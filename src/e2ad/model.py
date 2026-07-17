import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class ConvEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base_channels * 2), base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_channels * 2, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)


class GridAnatomicalTokenizer(nn.Module):
    """Atlas-free A-Tok approximation: splits feature maps into regular 3D ROI cells."""

    def __init__(self, roi_grid=(4, 4, 4)):
        super().__init__()
        self.roi_grid = tuple(roi_grid)

    def forward(self, fmap):
        pooled = F.adaptive_avg_pool3d(fmap, self.roi_grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        return F.layer_norm(tokens, (tokens.size(-1),))


class AnatomicalMixtureOfMappers(nn.Module):
    def __init__(self, num_rois, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.num_rois = num_rois
        self.specific = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
            for _ in range(num_rois)
        ])
        self.shared = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))
        self.reconstruct = nn.Sequential(nn.Linear(dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim))

    def forward(self, roi_features):
        specific = torch.stack([mapper(roi_features[:, idx]) for idx, mapper in enumerate(self.specific)], dim=1)
        shared = self.shared(roi_features)
        rec = self.reconstruct(torch.cat([shared, specific], dim=-1))
        return shared, specific, rec


class DifferentialAnatomicalRouter(nn.Module):
    def __init__(self, dim, heads=4, lambda_init=0.5):
        super().__init__()
        self.heads = heads
        self.query = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.SiLU(), nn.Conv1d(dim, dim * 2, 1))
        self.key = nn.Linear(dim, dim * 2)
        self.lambda_q1 = nn.Parameter(torch.randn(heads, dim) * 0.02)
        self.lambda_k1 = nn.Parameter(torch.randn(heads, dim) * 0.02)
        self.lambda_q2 = nn.Parameter(torch.randn(heads, dim) * 0.02)
        self.lambda_k2 = nn.Parameter(torch.randn(heads, dim) * 0.02)
        self.lambda_init = lambda_init

    def forward(self, roi_features):
        b, j, c = roi_features.shape
        pooled = roi_features.transpose(1, 2)
        q = self.query(pooled).mean(dim=-1).view(b, 2, c)
        k = self.key(roi_features).view(b, j, 2, c)
        attn1 = torch.softmax((q[:, 0:1] * k[:, :, 0]).sum(dim=-1) / math.sqrt(c), dim=-1)
        attn2 = torch.softmax((q[:, 1:2] * k[:, :, 1]).sum(dim=-1) / math.sqrt(c), dim=-1)
        lam = torch.exp((self.lambda_q1 * self.lambda_k1).sum(dim=1)).mean()
        lam = lam - torch.exp((self.lambda_q2 * self.lambda_k2).sum(dim=1)).mean() + self.lambda_init
        return torch.softmax(attn1 - lam * attn2, dim=-1)


class E2ADBackbone(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, roi_grid=(4, 4, 4), base_channels=16, feature_dim=64, heads=4):
        super().__init__()
        self.encoder = ConvEncoder3D(in_channels, base_channels, feature_dim)
        self.tokenizer = GridAnatomicalTokenizer(roi_grid)
        num_rois = int(roi_grid[0] * roi_grid[1] * roi_grid[2])
        self.mom = AnatomicalMixtureOfMappers(num_rois, feature_dim)
        self.router = DifferentialAnatomicalRouter(feature_dim, heads=heads)
        self.classifier = nn.Sequential(nn.LayerNorm(feature_dim * 3), nn.Linear(feature_dim * 3, feature_dim), nn.SiLU(), nn.Linear(feature_dim, num_classes))

    def forward(self, image, debug_shapes=False):
        fmap = self.encoder(image)
        roi = self.tokenizer(fmap)
        shared_roi, specific_roi, rec = self.mom(roi)
        weights = self.router(roi)
        f_shared = shared_roi.mean(dim=1)
        f_specific = torch.bmm(weights.unsqueeze(1), specific_roi).squeeze(1)
        f_global = fmap.mean(dim=(2, 3, 4))
        feature = torch.cat([f_shared, f_specific, f_global], dim=1)
        logits = self.classifier(feature)
        if debug_shapes:
            print("image:", image.shape, "fmap:", fmap.shape, "roi:", roi.shape)
            print("shared/specific:", shared_roi.shape, specific_roi.shape, "weights:", weights.shape)
            print("feature:", feature.shape, "logits:", logits.shape)
        return {
            "logits": logits,
            "feature": feature,
            "f_shared": f_shared,
            "f_specific": f_specific,
            "f_global": f_global,
            "roi": roi,
            "shared_roi": shared_roi,
            "specific_roi": specific_roi,
            "rec": rec,
            "weights": weights,
        }


class E2ADTeacher(nn.Module):
    def __init__(self, num_classes=2, roi_grid=(4, 4, 4), base_channels=16, feature_dim=64, heads=4):
        super().__init__()
        self.model = E2ADBackbone(2, num_classes, roi_grid, base_channels, feature_dim, heads)

    def forward(self, mri, pet, debug_shapes=False):
        return self.model(torch.cat([mri, pet], dim=1), debug_shapes=debug_shapes)


class E2ADStudent(nn.Module):
    def __init__(self, num_classes=2, roi_grid=(4, 4, 4), base_channels=16, feature_dim=64, heads=4):
        super().__init__()
        self.model = E2ADBackbone(1, num_classes, roi_grid, base_channels, feature_dim, heads)

    def forward(self, mri, debug_shapes=False):
        return self.model(mri, debug_shapes=debug_shapes)


class RelationProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shared = nn.Linear(dim, dim, bias=False)
        self.specific = nn.Linear(dim, dim, bias=False)

    def student_to_teacher(self, student_out):
        return torch.cat([self.shared(student_out["f_shared"]), self.specific(student_out["f_specific"])], dim=1)

    def teacher_to_student(self, teacher_out):
        return torch.cat([
            F.linear(teacher_out["f_shared"], self.shared.weight.t()),
            F.linear(teacher_out["f_specific"], self.specific.weight.t()),
        ], dim=1)


def mapper_regulation_loss(out):
    shared = out["shared_roi"]
    specific = out["specific_roi"]
    roi = out["roi"]
    rec = out["rec"]
    shared_center = shared.mean(dim=1, keepdim=True)
    shared_loss = (1.0 - F.cosine_similarity(shared, shared_center, dim=-1)).mean()
    spec_norm = F.normalize(specific, dim=-1)
    sim = torch.matmul(spec_norm, spec_norm.transpose(1, 2)).abs()
    eye = torch.eye(sim.size(1), device=sim.device, dtype=torch.bool).unsqueeze(0)
    inter = sim.masked_fill(eye, 0.0).sum() / max((~eye).sum().item() * sim.size(0), 1)
    intra = F.cosine_similarity(shared, specific, dim=-1).abs().mean()
    rec_loss = F.mse_loss(rec, roi)
    return shared_loss + inter + intra + rec_loss


def logit_kd_loss(teacher_logits, student_logits, temperature=2.0):
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student_log = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log, teacher_prob, reduction="batchmean") * (temperature ** 2)


def anatomy_kd_loss(teacher_weights, student_weights):
    teacher_prob = teacher_weights.detach().clamp_min(1e-6)
    student_log = student_weights.clamp_min(1e-6).log()
    return -(teacher_prob * student_log).sum(dim=1).mean()


def gram(features):
    features = F.normalize(features, dim=1)
    return features @ features.t()


def cka_loss(x, y):
    x = x - x.mean(dim=0, keepdim=True) - x.mean(dim=1, keepdim=True) + x.mean()
    y = y - y.mean(dim=0, keepdim=True) - y.mean(dim=1, keepdim=True) + y.mean()
    numerator = (x * y).sum().pow(2)
    denominator = (x * x).sum() * (y * y).sum() + 1e-6
    return 1.0 - numerator / denominator


def relation_kd_loss(teacher_out, student_out, projector):
    t_feat = teacher_out["feature"].detach()
    s_feat = student_out["feature"]
    r_tt = gram(t_feat)
    r_ss = gram(s_feat)
    s_to_t = projector.student_to_teacher(student_out)
    t_to_s = projector.teacher_to_student(teacher_out)
    r_st = gram(s_to_t)
    r_ts = gram(t_to_s)
    return cka_loss(r_ss, r_tt) + cka_loss(r_st, r_tt) + cka_loss(r_ts, r_tt)
