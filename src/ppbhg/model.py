import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class OrganEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, latent_channels=64):
        super().__init__()
        self.stem = ConvBlock3D(in_channels, base_channels, stride=2)
        self.stage1 = ConvBlock3D(base_channels, base_channels * 2, stride=2)
        self.stage2 = ConvBlock3D(base_channels * 2, latent_channels, stride=2)
        self.out_dim = latent_channels

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.stage1(f0)
        fmap = self.stage2(f1)
        pooled = fmap.mean(dim=(2, 3, 4))
        return fmap, pooled


class PositionalPromptBlock(nn.Module):
    """Approximate the paper's attention-map prompts with data-driven feature energy prompts."""

    def __init__(self, dim):
        super().__init__()
        self.prompt = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, brain_map):
        b, c, d, h, w = brain_map.shape
        tokens = brain_map.flatten(2).transpose(1, 2)
        energy = brain_map.abs().mean(dim=1, keepdim=True)
        local_mean = energy.flatten(2).transpose(1, 2)
        local_std = energy.flatten(2).std(dim=2, keepdim=True).transpose(1, 2).expand_as(local_mean)
        prompts = self.prompt(torch.cat([local_mean, local_std], dim=-1))
        return self.norm(tokens + prompts)


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query_tokens, context_tokens=None):
        if context_tokens is None:
            context_tokens = query_tokens
        attended, _ = self.attn(query_tokens, context_tokens, context_tokens, need_weights=False)
        x = self.norm1(query_tokens + attended)
        return self.norm2(x + self.ffn(x))


class PPBHGTeacher(nn.Module):
    def __init__(self, num_classes=2, base_channels=16, latent_channels=64, heads=4, dropout=0.0):
        super().__init__()
        self.brain = OrganEncoder3D(1, base_channels, latent_channels)
        self.heart = OrganEncoder3D(1, base_channels, latent_channels)
        self.gut = OrganEncoder3D(1, base_channels, latent_channels)
        self.prompt = PositionalPromptBlock(latent_channels)
        self.self_attn = CrossTransformerBlock(latent_channels, heads, dropout)
        self.brain_heart_attn = CrossTransformerBlock(latent_channels, heads, dropout)
        self.brain_gut_attn = CrossTransformerBlock(latent_channels, heads, dropout)
        self.heart_head = nn.Linear(latent_channels, num_classes)
        self.gut_head = nn.Linear(latent_channels, num_classes)
        self.bs_head = nn.Linear(latent_channels, num_classes)
        self.bh_head = nn.Linear(latent_channels, num_classes)
        self.bg_head = nn.Linear(latent_channels, num_classes)
        self.fusion = nn.Sequential(
            nn.LayerNorm(latent_channels * 3),
            nn.Linear(latent_channels * 3, latent_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(latent_channels, num_classes)

    def forward(self, brain_pet, heart, gut, debug_shapes=False):
        brain_map, brain_vec = self.brain(brain_pet)
        heart_map, heart_vec = self.heart(heart)
        gut_map, gut_vec = self.gut(gut)
        brain_tokens = self.prompt(brain_map)
        heart_tokens = heart_map.flatten(2).transpose(1, 2)
        gut_tokens = gut_map.flatten(2).transpose(1, 2)
        bs_tokens = self.self_attn(brain_tokens)
        bh_tokens = self.brain_heart_attn(brain_tokens, heart_tokens)
        bg_tokens = self.brain_gut_attn(brain_tokens, gut_tokens)
        f_bs = bs_tokens.mean(dim=1)
        f_bh = bh_tokens.mean(dim=1)
        f_bg = bg_tokens.mean(dim=1)
        fused = self.fusion(torch.cat([f_bs, f_bh, f_bg], dim=1))
        logits = self.classifier(fused)
        if debug_shapes:
            print("brain_pet:", brain_pet.shape, "heart:", heart.shape, "gut:", gut.shape)
            print("maps:", brain_map.shape, heart_map.shape, gut_map.shape)
            print("tokens:", brain_tokens.shape, heart_tokens.shape, gut_tokens.shape)
            print("F_bs/F_bh/F_bg:", f_bs.shape, f_bh.shape, f_bg.shape, "F_bhg:", fused.shape, "logits:", logits.shape)
        return {
            "brain_vec": brain_vec,
            "heart_vec": heart_vec,
            "gut_vec": gut_vec,
            "f_bs": f_bs,
            "f_bh": f_bh,
            "f_bg": f_bg,
            "f_bhg": fused,
            "heart_logits": self.heart_head(heart_vec),
            "gut_logits": self.gut_head(gut_vec),
            "bs_logits": self.bs_head(f_bs),
            "bh_logits": self.bh_head(f_bh),
            "bg_logits": self.bg_head(f_bg),
            "logits": logits,
        }


class PPBHGStudent(nn.Module):
    def __init__(self, num_classes=2, in_channels=1, base_channels=16, latent_channels=64, dropout=0.0):
        super().__init__()
        self.encoder = OrganEncoder3D(in_channels, base_channels, latent_channels)
        self.project = nn.Sequential(nn.LayerNorm(latent_channels), nn.Linear(latent_channels, latent_channels), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(latent_channels, num_classes))

    def forward(self, image, debug_shapes=False):
        fmap, pooled = self.encoder(image)
        feat = self.project(pooled)
        logits = self.classifier(feat)
        if debug_shapes:
            print("student_image:", image.shape, "student_map:", fmap.shape, "student_feat:", feat.shape, "student_logits:", logits.shape)
        return {"map": fmap, "feature": feat, "logits": logits}


def class_contrastive_loss(a, b, target, temperature=0.1):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.t() / temperature
    labels = target[:, None].eq(target[None, :]).float()
    labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def stage1_loss(out, target, lambda1=0.8, lambda2=1.2, lambda3=0.3, temperature=0.1):
    organ = F.cross_entropy(out["heart_logits"], target) + F.cross_entropy(out["gut_logits"], target)
    interaction = (
        F.cross_entropy(out["bs_logits"], target)
        + F.cross_entropy(out["bh_logits"], target)
        + F.cross_entropy(out["bg_logits"], target)
    )
    final = F.cross_entropy(out["logits"], target)
    cl = (
        class_contrastive_loss(out["brain_vec"], out["heart_vec"], target, temperature)
        + class_contrastive_loss(out["brain_vec"], out["gut_vec"], target, temperature)
        + class_contrastive_loss(out["f_bh"], out["f_bg"], target, temperature)
        + class_contrastive_loss(out["f_bg"], out["f_bh"], target, temperature)
    ) / 4.0
    return organ + lambda1 * interaction + lambda2 * final + lambda3 * cl


def sample_contrastive_distillation(teacher_feat, student_feat, temperature=0.1):
    teacher_feat = F.normalize(teacher_feat.detach(), dim=1)
    student_feat = F.normalize(student_feat, dim=1)
    logits = student_feat @ teacher_feat.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def group_distribution_distillation(teacher_feat, student_feat, target):
    losses = []
    for cls in target.unique():
        mask = target == cls
        if mask.sum() < 2:
            continue
        t = teacher_feat.detach()[mask]
        s = student_feat[mask]
        t_center = t.mean(dim=0, keepdim=True)
        s_center = s.mean(dim=0, keepdim=True)
        t_var = F.cosine_similarity(t, t_center, dim=1)
        s_var = F.cosine_similarity(s, s_center, dim=1)
        losses.append(F.mse_loss(s_var, t_var))
    if not losses:
        return student_feat.new_tensor(0.0)
    return torch.stack(losses).mean()


def response_kd_loss(teacher_logits, student_logits, temperature=2.0):
    t = F.softmax(teacher_logits.detach() / temperature, dim=1)
    s = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)

