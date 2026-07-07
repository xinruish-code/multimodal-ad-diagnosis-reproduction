import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CNNBackbone3D(nn.Module):
    def __init__(self, channels=32, blocks=4):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(blocks):
            layers.append(ConvBlock3D(in_channels, channels))
            in_channels = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AnatomyAwareSE(nn.Module):
    def __init__(self, channels, num_rois, embed_dim=128, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.excite = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, channels),
                    nn.Sigmoid(),
                )
                for _ in range(num_rois)
            ]
        )
        self.embed = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_rois)
            ]
        )

    def forward(self, feature_map, roi_masks):
        b, c, d, h, w = feature_map.shape
        masks = F.interpolate(roi_masks[:, None], size=(d, h, w), mode="nearest").squeeze(1)
        masks = masks.to(device=feature_map.device, dtype=feature_map.dtype)
        flat_feat = feature_map.flatten(2)
        flat_masks = masks.flatten(1)
        denom = flat_masks.sum(dim=1).clamp_min(1.0)
        squeezed = torch.einsum("bcn,kn->bkc", flat_feat, flat_masks) / denom.view(1, -1, 1)
        roi_embeddings = []
        for idx, (excite, embed) in enumerate(zip(self.excite, self.embed)):
            sc = squeezed[:, idx]
            roi_embeddings.append(embed(excite(sc) * sc))
        return torch.stack(roi_embeddings, dim=1), squeezed


class AnatomyGating(nn.Module):
    def __init__(self, embed_dim, num_rois, tau=1.0, hard=False):
        super().__init__()
        self.num_rois = num_rois
        self.tau = tau
        self.hard = hard
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_rois),
        )

    @staticmethod
    def _sample_gumbel(shape, device):
        u = torch.rand(shape, device=device).clamp_(1e-6, 1 - 1e-6)
        return -torch.log(-torch.log(u))

    def forward(self, roi_embeddings, training=True):
        context = torch.cat([roi_embeddings.mean(dim=1), roi_embeddings.max(dim=1).values], dim=-1)
        logits = self.net(context)
        if training:
            g1 = self._sample_gumbel(logits.shape, logits.device)
            g2 = self._sample_gumbel(logits.shape, logits.device)
            z_soft = torch.sigmoid((logits + g1 - g2) / self.tau)
        else:
            z_soft = torch.sigmoid(logits)
        if self.hard:
            z_hard = (z_soft > 0.5).to(z_soft.dtype)
            z = z_hard.detach() - z_soft.detach() + z_soft
        else:
            z = z_soft
        return z, z_soft, logits


class AAGN(nn.Module):
    def __init__(self, roi_masks, num_classes=2, channels=32, blocks=4, embed_dim=128, tau=1.0, hard=False):
        super().__init__()
        self.register_buffer("roi_masks", roi_masks.float())
        self.backbone = CNNBackbone3D(channels, blocks)
        self.roi_se = AnatomyAwareSE(channels, roi_masks.size(0), embed_dim)
        self.gate = AnatomyGating(embed_dim, roi_masks.size(0), tau=tau, hard=hard)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, image, debug_shapes=False):
        feat_map = self.backbone(image)
        roi_embeddings, roi_channel_features = self.roi_se(feat_map, self.roi_masks)
        z, z_soft, gate_logits = self.gate(roi_embeddings, self.training)
        selected = (roi_embeddings * z.unsqueeze(-1)).sum(dim=1) / z.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = self.classifier(selected)
        if debug_shapes:
            print("image:", image.shape)
            print("feat_map:", feat_map.shape)
            print("roi_masks:", self.roi_masks.shape)
            print("roi_embeddings:", roi_embeddings.shape)
            print("gate:", z.shape, "selected:", selected.shape, "logits:", logits.shape)
        return {
            "logits": logits,
            "features": selected,
            "roi_embeddings": roi_embeddings,
            "roi_channel_features": roi_channel_features,
            "gate": z,
            "gate_prob": z_soft,
            "gate_logits": gate_logits,
        }
