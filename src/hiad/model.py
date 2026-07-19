import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class ResConv3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class SerpentineHilbertLikeOrder:
    """A lightweight locality-preserving scan used as a Hilbert-curve approximation."""

    @staticmethod
    def indices(d, h, w, view, device):
        coords = []
        for z in range(d):
            y_range = range(h) if z % 2 == 0 else range(h - 1, -1, -1)
            for y in y_range:
                x_range = range(w) if (y + z) % 2 == 0 else range(w - 1, -1, -1)
                for x in x_range:
                    if view == 0:
                        coords.append((z, y, x))
                    elif view == 1:
                        coords.append((z, x, y))
                    else:
                        coords.append((y, z, x))
        coords = [(min(a, d - 1), min(b, h - 1), min(c, w - 1)) for a, b, c in coords]
        flat = torch.tensor([a * h * w + b * w + c for a, b, c in coords], device=device, dtype=torch.long)
        return flat.unique(sorted=False)[: d * h * w]


class SequenceMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mix = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, seq):
        x = self.norm(seq)
        y = self.mix(x.transpose(1, 2)).transpose(1, 2)
        return seq + y


class HMambaApproxBlock(nn.Module):
    def __init__(self, channels, views=3):
        super().__init__()
        self.views = int(views)
        self.mixers = nn.ModuleList([SequenceMixer(channels) for _ in range(self.views)])
        self.view_score = nn.Conv3d(channels, 1, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        flat = x.flatten(2).transpose(1, 2)
        view_maps = []
        scores = []
        for view in range(self.views):
            order = SerpentineHilbertLikeOrder.indices(d, h, w, view, x.device)
            if order.numel() < d * h * w:
                order = torch.arange(d * h * w, device=x.device)
            seq = flat[:, order]
            seq = self.mixers[view](seq)
            inv = torch.empty_like(order)
            inv[torch.arange(order.numel(), device=x.device)] = torch.arange(order.numel(), device=x.device)
            restored = torch.zeros_like(flat)
            restored[:, order] = seq[:, inv]
            fmap = restored.transpose(1, 2).reshape(b, c, d, h, w)
            view_maps.append(fmap)
            scores.append(self.view_score(fmap))
        weights = torch.softmax(torch.stack(scores, dim=1), dim=1)
        maps = torch.stack(view_maps, dim=1)
        return (weights * maps).sum(dim=1)


class HSFE(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, feature_dim=128, views=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.GELU(),
        )
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.downs = nn.ModuleList([
            nn.Identity(),
            nn.Conv3d(channels[0], channels[1], 3, stride=2, padding=1, bias=False),
            nn.Conv3d(channels[1], channels[2], 3, stride=2, padding=1, bias=False),
        ])
        self.blocks = nn.ModuleList([HMambaApproxBlock(ch, views) for ch in channels])
        self.res = nn.ModuleList([ResConv3D(ch) for ch in channels])
        self.proj = nn.ModuleList([nn.Linear(ch, feature_dim) for ch in channels])
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x, debug_shapes=False):
        x = self.stem(x)
        outputs = []
        for idx, (down, block, res, proj) in enumerate(zip(self.downs, self.blocks, self.res, self.proj)):
            x = down(x)
            x = block(x)
            x = res(x)
            outputs.append(proj(x.mean(dim=(2, 3, 4))))
        feat = self.norm(torch.stack(outputs, dim=0).max(dim=0).values)
        if debug_shapes:
            print("hsfe_levels:", [o.shape for o in outputs], "feat:", feat.shape)
        return feat


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class HIAD(nn.Module):
    def __init__(self, num_classes=2, cad_dim=1, base_channels=16, feature_dim=128, views=3, beta=0.97, tau=0.2):
        super().__init__()
        self.beta = beta
        self.tau = tau
        self.mri = HSFE(1, base_channels, feature_dim, views)
        self.pet = HSFE(1, base_channels, feature_dim, views)
        self.cad = TabularEncoder(cad_dim, feature_dim)
        self.posteriors = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])
        self.decoders = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(3)])
        self.confidence = nn.ModuleList([nn.Sequential(nn.Linear(feature_dim, feature_dim // 2), nn.GELU(), nn.Linear(feature_dim // 2, 1), nn.Sigmoid()) for _ in range(3)])
        self.uni_heads = nn.ModuleList([nn.Linear(feature_dim, num_classes) for _ in range(3)])
        self.shared_head = nn.Linear(feature_dim, num_classes)

    def forward(self, mri, pet, cad, rho, epoch=1, train_mode=True, debug_shapes=False):
        feats = [self.mri(mri, debug_shapes=debug_shapes), self.pet(pet), self.cad(cad)]
        z_list = [proj(feat) for proj, feat in zip(self.posteriors, feats)]
        eta = torch.cat([conf(z).clamp_min(1e-4) for conf, z in zip(self.confidence, z_list)], dim=1)
        uni_logits = [head(feat) for head, feat in zip(self.uni_heads, feats)]
        conf_pred = torch.stack([torch.softmax(logit, dim=1).max(dim=1).values for logit in uni_logits], dim=1)
        sign = -1.0 if train_mode else 1.0
        mid_scores = sign * eta / self.tau
        mid_scores = mid_scores.masked_fill(rho <= 0, -1e4)
        omega = torch.softmax(mid_scores, dim=1)
        z = sum(omega[:, idx:idx + 1] * z_list[idx] for idx in range(3))
        shared_logits = self.shared_head(z)
        late_scores = sign * eta / self.tau + conf_pred.detach().clamp_min(1e-4).log()
        late_scores = late_scores.masked_fill(rho <= 0, -1e4)
        late_w = torch.softmax(late_scores, dim=1)
        uni_prob = sum(late_w[:, idx:idx + 1] * torch.softmax(uni_logits[idx], dim=1) for idx in range(3))
        beta_t = self.beta ** max(epoch, 1)
        prob = (1.0 - beta_t) * torch.softmax(shared_logits, dim=1) + beta_t * uni_prob
        logits = (prob.clamp_min(1e-6)).log()
        if debug_shapes:
            print("modal_feats:", [f.shape for f in feats], "rho:", rho.shape, "omega:", omega.shape, "logits:", logits.shape)
        return {
            "logits": logits,
            "prob": prob,
            "features": feats,
            "z": z,
            "z_list": z_list,
            "eta": eta,
            "omega": omega,
            "late_weights": late_w,
            "uni_logits": uni_logits,
        }


def hiad_losses(out, target, rho, lambda_intra=1.0, gamma_inter=1.0):
    ce = F.nll_loss(out["logits"], target)
    uni_losses = torch.stack([F.cross_entropy(logit, target, reduction="none") for logit in out["uni_logits"]], dim=1)
    b = torch.softmax(-uni_losses.detach().masked_fill(rho <= 0, 1e4), dim=1)
    conf_loss = (((out["eta"] - b) ** 2) * rho).sum() / rho.sum().clamp_min(1.0)
    intra = 0.0
    inter = 0.0
    count_inter = 0
    for idx, decoder in enumerate(out["decoders"] if "decoders" in out else []):
        pass
    for idx, (feat, decoder) in enumerate(zip(out["features"], [])):
        pass
    # Reconstruction heads are stored on the model; this helper handles alignment losses only.
    z_list = out["z_list"]
    feats = out["features"]
    for idx in range(3):
        intra = intra + F.mse_loss(z_list[idx], feats[idx].detach(), reduction="none").mean(dim=1).mul(rho[:, idx]).sum() / rho[:, idx].sum().clamp_min(1.0)
        for jdx in range(3):
            if idx == jdx:
                continue
            mask = rho[:, idx] * rho[:, jdx]
            inter = inter + F.mse_loss(z_list[idx], feats[jdx].detach(), reduction="none").mean(dim=1).mul(mask).sum() / mask.sum().clamp_min(1.0)
            count_inter += 1
    inter = inter / max(count_inter, 1)
    return ce + lambda_intra * intra + gamma_inter * inter + conf_loss
