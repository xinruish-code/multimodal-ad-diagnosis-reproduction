import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out + identity, inplace=True)


class ResNet18Branch3D(nn.Module):
    def __init__(self, in_channels=1, channels=(16, 32, 64, 128)):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(channels[0], channels[0], 2, stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], 2, stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], 2, stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], 2, stride=2)

    @staticmethod
    def _make_layer(in_channels, out_channels, blocks, stride):
        layers = [BasicBlock3D(in_channels, out_channels, stride)]
        layers += [BasicBlock3D(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)
        return h1, h2, h3, h4


class SATFusion(nn.Module):
    def __init__(self, channels=128, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.wa = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv3d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    @staticmethod
    def _tokens(x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, pet, gm, wm):
        tp, tg, tw = self._tokens(pet), self._tokens(gm), self._tokens(wm)
        hp = self.attn(tp, tp, tp, need_weights=False)[0]
        hg = self.attn(tg, tg, tg, need_weights=False)[0]
        hw = self.attn(tw, tw, tw, need_weights=False)[0]
        gate = self.wa(pet + gm + wm).flatten(2).transpose(1, 2)
        fused = (hp + hg + hw) * gate
        return self.mlp(fused).mean(dim=1)


class LALFusion(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        self.proj1 = nn.Conv3d(channels[0], channels[-1], 1)
        self.proj2 = nn.Conv3d(channels[1], channels[-1], 1)
        self.proj3 = nn.Conv3d(channels[2], channels[-1], 1)
        self.conv3 = nn.Conv3d(channels[-1], channels[-1], 3, padding=1)
        self.conv1 = nn.Conv3d(channels[-1], channels[-1], 1)

    def _resize(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, pet_scales, gm_scales, wm_scales):
        h1 = pet_scales[0] + gm_scales[0] + wm_scales[0]
        h2 = pet_scales[1] + gm_scales[1] + wm_scales[1]
        h3 = pet_scales[2] + gm_scales[2] + wm_scales[2]
        h2 = self._resize(h2, h3)
        h1 = self._resize(h1, h3)
        x = self.proj1(h1) + self.proj2(h2) + self.proj3(h3)
        x = torch.sigmoid(self.conv3(x)) * x
        x = self.conv1(x)
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


class LSLFusion(nn.Module):
    def __init__(self, channels=128, pooled_tokens=8):
        super().__init__()
        self.pooled_tokens = pooled_tokens
        self.proj = nn.Linear(pooled_tokens * pooled_tokens, 1)
        self.norm = nn.LayerNorm(channels)

    def _pool(self, x):
        avg = F.adaptive_avg_pool3d(x, (2, 2, 2)).flatten(2)
        mx = F.adaptive_max_pool3d(x, (2, 2, 2)).flatten(2)
        return 0.5 * (avg + mx)

    def forward(self, pet, gm, wm):
        p, g, w = self._pool(pet), self._pool(gm), self._pool(wm)
        outer = (
            torch.einsum("bct,bcu->bctu", p, g)
            + torch.einsum("bct,bcu->bctu", p, w)
            + torch.einsum("bct,bcu->bctu", g, w)
        )
        outer = outer.flatten(2)
        return self.norm(F.normalize(self.proj(outer).squeeze(-1), p=2, dim=1))


class DRLHead(nn.Module):
    def __init__(self, channels=128, roi_dim=90, iterations=3):
        super().__init__()
        self.iterations = iterations
        self.cls_to_roi = nn.Linear(2, roi_dim)
        self.roi_seed = nn.Linear(channels, roi_dim)
        self.gru = nn.GRUCell(roi_dim, roi_dim)
        self.roi_out = nn.Linear(roi_dim, roi_dim)
        self.roi_cls = nn.Linear(roi_dim, 2)

    def forward(self, features, cls_logits):
        roi = self.roi_seed(features)
        hidden = torch.sigmoid(self.cls_to_roi(cls_logits))
        for _ in range(self.iterations):
            weighted = roi * torch.sigmoid(hidden)
            hidden = self.gru(weighted, hidden)
        roi_pred = self.roi_out(hidden)
        logits = cls_logits + self.roi_cls(roi_pred)
        return logits, roi_pred


class MDLNet(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), num_classes=2, use_drl=False, drl_iterations=3):
        super().__init__()
        self.pet_branch = ResNet18Branch3D(1, channels)
        self.gm_branch = ResNet18Branch3D(1, channels)
        self.wm_branch = ResNet18Branch3D(1, channels)
        c4 = channels[-1]
        self.sat = SATFusion(c4)
        self.lal = LALFusion(channels)
        self.lsl = LSLFusion(c4)
        self.fuse = nn.Sequential(nn.Linear(c4 * 3, c4), nn.LayerNorm(c4), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(c4, num_classes)
        self.use_drl = use_drl
        self.drl = DRLHead(c4, iterations=drl_iterations) if use_drl else None

    def forward(self, image, debug_shapes=False):
        pet = image[:, 0:1]
        gm = image[:, 1:2]
        wm = image[:, 2:3]
        hp = self.pet_branch(pet)
        hg = self.gm_branch(gm)
        hw = self.wm_branch(wm)
        fs = self.sat(hp[-1], hg[-1], hw[-1])
        fl = self.lal(hp[:3], hg[:3], hw[:3])
        fo = self.lsl(hp[-1], hg[-1], hw[-1])
        cat = torch.cat([fs, fl, fo], dim=1)
        feat = self.fuse(cat)
        logits = self.classifier(feat)
        if debug_shapes:
            print("image:", image.shape)
            print("pet:", pet.shape, "gm:", gm.shape, "wm:", wm.shape)
            print("hp:", [x.shape for x in hp])
            print("hg:", [x.shape for x in hg])
            print("hw:", [x.shape for x in hw])
            print("fs:", fs.shape)
            print("fl:", fl.shape)
            print("fo:", fo.shape)
            print("cat:", cat.shape)
            print("feat:", feat.shape)
            print("logits:", logits.shape)

        out = {"logits": logits, "features": feat}
        if self.use_drl:
            logits, roi_pred = self.drl(feat, logits)
            out["logits"] = logits
            out["roi_pred"] = roi_pred
        return out
