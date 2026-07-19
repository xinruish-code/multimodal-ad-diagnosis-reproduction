import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class ConvStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class VGG3DEncoder(nn.Module):
    def __init__(self, in_channels=1, channels=(16, 32, 64), feature_dim=128):
        super().__init__()
        self.stages = nn.ModuleList()
        cur = in_channels
        for ch in channels:
            self.stages.append(ConvStage(cur, ch))
            cur = ch
        self.project = nn.Sequential(nn.Linear(channels[-1], feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
        self.out_dim = feature_dim

    def forward(self, x):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        high = self.project(x.mean(dim=(2, 3, 4)))
        return feats, high


class CRADTeacher(nn.Module):
    def __init__(self, num_classes=2, channels=(16, 32, 64), feature_dim=128):
        super().__init__()
        self.mri_encoder = VGG3DEncoder(1, channels, feature_dim)
        self.pet_encoder = VGG3DEncoder(1, channels, feature_dim)
        self.mmse_heads = nn.ModuleList([nn.Linear(ch, 1) for ch in channels])
        self.fuse = nn.Sequential(nn.LayerNorm(feature_dim * 2), nn.Linear(feature_dim * 2, feature_dim), nn.GELU())
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, mri, pet, debug_shapes=False):
        mri_feats, mri_high = self.mri_encoder(mri)
        pet_feats, pet_high = self.pet_encoder(pet)
        fused = self.fuse(torch.cat([mri_high, pet_high], dim=1))
        logits = self.classifier(fused)
        mmse_pred = [head(feat.mean(dim=(2, 3, 4))).squeeze(1) for head, feat in zip(self.mmse_heads, mri_feats)]
        if debug_shapes:
            print("teacher mri/pet:", mri.shape, pet.shape)
            print("teacher feats:", [f.shape for f in mri_feats], "high:", fused.shape, "logits:", logits.shape)
        return {
            "logits": logits,
            "feature": fused,
            "mri_high": mri_high,
            "pet_high": pet_high,
            "mri_feats": mri_feats,
            "pet_feats": pet_feats,
            "mmse_pred": mmse_pred,
        }


class CRADStudent(nn.Module):
    def __init__(self, num_classes=2, channels=(16, 32, 64), feature_dim=128):
        super().__init__()
        self.encoder = VGG3DEncoder(1, channels, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, mri, debug_shapes=False):
        feats, high = self.encoder(mri)
        logits = self.classifier(high)
        if debug_shapes:
            print("student mri:", mri.shape)
            print("student feats:", [f.shape for f in feats], "high:", high.shape, "logits:", logits.shape)
        return {"logits": logits, "feature": high, "mri_feats": feats}


def orthogonal_loss(teacher_out):
    m = F.normalize(teacher_out["mri_high"], dim=1)
    p = F.normalize(teacher_out["pet_high"], dim=1)
    same = 1.0 - (m * m).sum(dim=1).mean()
    cross = (m * p).sum(dim=1).abs().mean()
    return same + cross


def mmse_loss(teacher_out, mmse):
    losses = [F.mse_loss(pred, mmse) for pred in teacher_out["mmse_pred"]]
    return torch.stack(losses).mean()


def relation_kd_loss(teacher_feat, student_feat, temperature=1.0):
    t = F.normalize(teacher_feat.detach(), dim=1)
    s = F.normalize(student_feat, dim=1)
    tt = F.softmax((t @ t.t()) / temperature, dim=1)
    ss = F.log_softmax((s @ s.t()) / temperature, dim=1)
    return F.kl_div(ss, tt, reduction="batchmean")


def soft_label_kd_loss(teacher_logits, student_logits, temperature=2.0):
    t = F.softmax(teacher_logits.detach() / temperature, dim=1)
    s = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def pfac_attention(feat):
    b, c, d, h, w = feat.shape
    x = F.normalize(feat.flatten(2), dim=1)
    gap = x.mean(dim=2)
    gmp = x.max(dim=2).values
    score = torch.sigmoid((gap + gmp).view(b, c, 1, 1, 1))
    return score


def pf_cmad_loss(teacher_out, student_out):
    losses = []
    for tf, sf in zip(teacher_out["mri_feats"], student_out["mri_feats"]):
        if tf.shape[2:] != sf.shape[2:]:
            sf = F.interpolate(sf, size=tf.shape[2:], mode="trilinear", align_corners=False)
        at = pfac_attention(tf.detach())
        ast = pfac_attention(sf)
        losses.append(F.mse_loss(ast * sf, at * tf.detach()))
    return torch.stack(losses).mean()


def smooth_confidence(teacher_logits, student_logits, target, eps=1e-6):
    teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
    student_prob = torch.softmax(student_logits.detach(), dim=1)
    y = F.one_hot(target, teacher_prob.size(1)).float()
    teacher_err = ((teacher_prob - y) ** 2).sum(dim=1)
    student_err = ((student_prob - y) ** 2).sum(dim=1)
    sc = 1.0 - teacher_err.clamp(0, 1)
    psi = (student_err / (teacher_err + student_err + eps)).pow(2)
    return (sc * psi).detach().mean()

