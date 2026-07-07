import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchViT3D(nn.Module):
    def __init__(self, patch_size=8, embed_dim=128, depth=2, heads=4, dropout=0.0, max_tokens=4096):
        super().__init__()
        self.patch = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos[:, : tokens.size(1)]
        tokens = self.encoder(tokens)
        return self.norm(tokens.mean(dim=1))


class MetaEncoder(nn.Module):
    def __init__(self, meta_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, meta):
        return self.net(meta)


class UniCross(nn.Module):
    def __init__(
        self,
        meta_dim,
        num_classes=2,
        patch_size=8,
        embed_dim=128,
        depth=2,
        heads=4,
        dropout=0.0,
        max_tokens=4096,
    ):
        super().__init__()
        self.mri_encoder = PatchViT3D(patch_size, embed_dim, depth, heads, dropout, max_tokens)
        self.pet_encoder = PatchViT3D(patch_size, embed_dim, depth, heads, dropout, max_tokens)
        self.meta_encoder = MetaEncoder(meta_dim, embed_dim)
        self.mri_head = nn.Linear(embed_dim, num_classes)
        self.pet_head = nn.Linear(embed_dim, num_classes)
        self.shared_head = nn.Linear(embed_dim, num_classes)
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def encode(self, mri, pet, meta):
        f_mri = self.mri_encoder(mri)
        f_pet = self.pet_encoder(pet)
        f_meta = self.meta_encoder(meta)
        return f_mri, f_pet, f_meta

    def forward_encoder_stage(self, mri, pet, meta, debug_shapes=False):
        f_mri, f_pet, f_meta = self.encode(mri, pet, meta)
        out = {
            "f_mri": f_mri,
            "f_pet": f_pet,
            "f_meta": f_meta,
            "mri_logits": self.mri_head(f_mri),
            "pet_logits": self.pet_head(f_pet),
            "shared_mri_logits": self.shared_head(f_mri),
            "shared_pet_logits": self.shared_head(f_pet),
        }
        if debug_shapes:
            print("f_mri:", f_mri.shape, "f_pet:", f_pet.shape, "f_meta:", f_meta.shape)
            print("mri_logits:", out["mri_logits"].shape, "pet_logits:", out["pet_logits"].shape)
        return out

    def forward_fusion_stage(self, mri, pet, meta, debug_shapes=False):
        f_mri, f_pet, f_meta = self.encode(mri, pet, meta)
        fused = torch.cat([f_mri, f_pet], dim=1)
        logits = self.fusion_head(fused)
        if debug_shapes:
            print("f_mri:", f_mri.shape, "f_pet:", f_pet.shape, "f_meta:", f_meta.shape)
            print("fused:", fused.shape, "logits:", logits.shape)
        return {"logits": logits, "features": fused, "f_mri": f_mri, "f_pet": f_pet, "f_meta": f_meta}


def metadata_weighted_contrastive(f_mri, f_pet, f_meta, target, temperature=0.07):
    feats = {
        "mri": F.normalize(f_mri, dim=1),
        "pet": F.normalize(f_pet, dim=1),
    }
    meta = F.normalize(f_meta, dim=1)
    label_mask = (target[:, None] == target[None, :]).float()
    meta_sim = ((meta @ meta.t()) + 1.0) * 0.5
    weights = torch.nan_to_num(label_mask * meta_sim, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = weights.sum(dim=1, keepdim=True)
    fallback = torch.eye(weights.size(0), device=weights.device, dtype=weights.dtype)
    weights = torch.where(row_sum > 1e-6, weights / row_sum.clamp_min(1e-6), fallback)
    loss = f_mri.new_tensor(0.0)
    count = 0
    for anchor_name in ("mri", "pet"):
        for positive_name in ("mri", "pet"):
            logits = feats[anchor_name] @ feats[positive_name].t() / temperature
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss = loss - (weights * log_prob).sum(dim=1).mean()
            count += 1
    return loss / count
