import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenTransformer3D(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, d, h, w)


class PanSwinStage(nn.Module):
    """A compact PanSwin-inspired global compression block."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0):
        super().__init__()
        self.global_block = TokenTransformer3D(in_dim, heads=heads, dropout=dropout)
        self.primary = nn.Sequential(
            nn.AvgPool3d(2, ceil_mode=True),
            nn.Conv3d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm3d(out_dim),
        )
        self.tgic = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm3d(out_dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = self.global_block(x)
        primary = self.primary(x)
        global_context = self.tgic(x).expand_as(primary)
        return self.act(primary + global_context)


class PanSwinEncoder3D(nn.Module):
    def __init__(self, patch_size=8, embed_dim=32, heads=(2, 4, 4), dropout=0.0):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
        )
        self.stage1 = PanSwinStage(embed_dim, embed_dim * 2, heads=heads[0], dropout=dropout)
        self.stage2 = PanSwinStage(embed_dim * 2, embed_dim * 4, heads=heads[1], dropout=dropout)
        self.out_dim = embed_dim * 4

    def forward(self, x):
        f0 = self.patch(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        pooled = [f.mean(dim=(2, 3, 4)) for f in [f0, f1, f2]]
        return pooled, pooled[-1]


class MOGADTeacher(nn.Module):
    def __init__(self, num_classes=2, patch_size=8, embed_dim=32, dropout=0.0):
        super().__init__()
        self.brain = PanSwinEncoder3D(patch_size, embed_dim, dropout=dropout)
        self.aux1 = PanSwinEncoder3D(patch_size, embed_dim, dropout=dropout)
        self.aux2 = PanSwinEncoder3D(patch_size, embed_dim, dropout=dropout)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4]
        self.brain_head = nn.Linear(dims[-1], num_classes)
        self.aux1_head = nn.Linear(dims[-1], num_classes)
        self.aux2_head = nn.Linear(dims[-1], num_classes)
        self.fusion = nn.Sequential(
            nn.LayerNorm(dims[-1] * 3),
            nn.Linear(dims[-1] * 3, dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, brain_pet, aux1, aux2, debug_shapes=False):
        fb, zb = self.brain(brain_pet)
        fh, zh = self.aux1(aux1)
        fg, zg = self.aux2(aux2)
        fused = torch.cat([zb, zh, zg], dim=1)
        logits = self.fusion(fused)
        if debug_shapes:
            print("brain_pet:", brain_pet.shape, "aux1:", aux1.shape, "aux2:", aux2.shape)
            print("teacher final:", zb.shape, zh.shape, zg.shape, "fused:", fused.shape, "logits:", logits.shape)
        return {
            "features": [fb, fh, fg],
            "finals": [zb, zh, zg],
            "brain_logits": self.brain_head(zb),
            "aux1_logits": self.aux1_head(zh),
            "aux2_logits": self.aux2_head(zg),
            "logits": logits,
            "fused": fused,
        }


class MOGADStudent(nn.Module):
    def __init__(self, num_classes=2, patch_size=8, embed_dim=32, dropout=0.0):
        super().__init__()
        self.encoder = PanSwinEncoder3D(patch_size, embed_dim, dropout=dropout)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4]
        self.proj = nn.ModuleList([nn.Linear(dim, dim * 3) for dim in dims])
        self.head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, image, debug_shapes=False):
        features, final = self.encoder(image)
        logits = self.head(final)
        if debug_shapes:
            print("student_image:", image.shape)
            print("student_features:", [f.shape for f in features], "logits:", logits.shape)
        return {"features": features, "final": final, "logits": logits}


def hfa_loss(teacher_out, temperature=0.1):
    brain, aux1, aux2 = teacher_out["features"]
    loss = brain[0].new_tensor(0.0)
    for fb, fh, fg in zip(brain, aux1, aux2):
        loss = loss + (1.0 - F.cosine_similarity(fb, fh, dim=1)).mean() / temperature
        loss = loss + (1.0 - F.cosine_similarity(fb, fg, dim=1)).mean() / temperature
    return loss / len(brain)


def label_consistency_loss(teacher_out, target, temperature=0.1):
    organ_features = teacher_out["features"]
    loss = organ_features[0][0].new_tensor(0.0)
    count = 0
    for fb, fh, fg in zip(*organ_features):
        feat = F.normalize(torch.cat([fb, fh, fg], dim=1), dim=1)
        logits = feat @ feat.t() / temperature
        labels = (target[:, None] == target[None, :]).float()
        eye = torch.eye(labels.size(0), device=labels.device)
        labels = labels * (1.0 - eye)
        denom = torch.logsumexp(logits.masked_fill(eye.bool(), -1e9), dim=1)
        pos = torch.logsumexp(logits.masked_fill(labels <= 0, -1e9), dim=1)
        valid = labels.sum(dim=1) > 0
        if valid.any():
            loss = loss - (pos[valid] - denom[valid]).mean()
            count += 1
    return loss / max(count, 1)


def hkd_loss(student_out, teacher_out, student_model, temperature=0.1):
    brain, aux1, aux2 = teacher_out["features"]
    loss = student_out["features"][0].new_tensor(0.0)
    for idx, (fs, fb, fh, fg) in enumerate(zip(student_out["features"], brain, aux1, aux2)):
        teacher_fused = torch.cat([fb, fh, fg], dim=1).detach()
        student_proj = student_model.proj[idx](fs)
        loss = loss + (1.0 - F.cosine_similarity(student_proj, teacher_fused, dim=1)).mean() / temperature
    return loss / len(student_out["features"])
