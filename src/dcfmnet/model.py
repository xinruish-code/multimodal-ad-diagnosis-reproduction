import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class ConvEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.GELU(),
            nn.MaxPool3d(3, stride=2, padding=1),
            nn.Conv3d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base_channels * 2), base_channels * 2),
            nn.GELU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_groups(base_channels * 4), base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        hidden = base_channels * 4
        self.common = nn.Sequential(nn.Flatten(), nn.Linear(hidden, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
        self.specific = nn.Sequential(nn.Flatten(), nn.Linear(hidden, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())

    def forward(self, x):
        base = self.conv(x)
        return self.common(base), self.specific(base)


class FeatureDecoder(nn.Module):
    """Compact reconstruction head for the paper's self/cross reconstruction losses."""

    def __init__(self, feature_dim=128, recon_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, recon_dim),
        )

    def forward(self, common, specific):
        return self.net(torch.cat([common, specific], dim=1))


class FeatureCodebook(nn.Module):
    def __init__(self, codebook_size=64, feature_dim=128):
        super().__init__()
        self.codebook = nn.Parameter(torch.empty(codebook_size, feature_dim))
        nn.init.uniform_(self.codebook, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, features):
        distances = torch.cdist(features, self.codebook)
        indices = distances.argmin(dim=1)
        quantized = self.codebook[indices]
        codebook_loss = F.mse_loss(quantized, features.detach())
        quantized_st = features + (quantized - features).detach()
        return quantized_st, codebook_loss, indices

    def nearest_from_query(self, query):
        distances = torch.cdist(query, self.codebook)
        indices = distances.argmin(dim=1)
        return self.codebook[indices], indices


class SimilarityDistinguisher(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, left, right):
        return self.net(torch.cat([left, right], dim=-1)).squeeze(-1)


class DCFMNet(nn.Module):
    def __init__(self, num_classes=2, base_channels=16, feature_dim=128, codebook_size=64):
        super().__init__()
        self.mri_encoder = ConvEncoder3D(1, base_channels, feature_dim)
        self.pet_encoder = ConvEncoder3D(1, base_channels, feature_dim)
        self.mri_decoder = FeatureDecoder(feature_dim, feature_dim)
        self.pet_decoder = FeatureDecoder(feature_dim, feature_dim)
        self.mri_common_codebook = FeatureCodebook(codebook_size, feature_dim)
        self.mri_specific_codebook = FeatureCodebook(codebook_size, feature_dim)
        self.pet_common_codebook = FeatureCodebook(codebook_size, feature_dim)
        self.pet_specific_codebook = FeatureCodebook(codebook_size, feature_dim)
        self.mri_to_pet_common = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        self.mri_to_pet_specific = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
        self.common_matcher = SimilarityDistinguisher(feature_dim)
        self.specific_matcher = SimilarityDistinguisher(feature_dim)
        self.common_weights = nn.Parameter(torch.ones(2))
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_classes),
        )

    def _fuse(self, c_mri, s_mri, c_pet, s_pet):
        weights = torch.softmax(self.common_weights, dim=0)
        common = weights[0] * c_mri + weights[1] * c_pet
        return torch.cat([common, s_mri, s_pet], dim=1)

    def forward(self, mri, pet=None, missing_pet=False, debug_shapes=False):
        mri_com, mri_spe = self.mri_encoder(mri)
        c1, cb_c1, idx_c1 = self.mri_common_codebook(mri_com)
        s1, cb_s1, idx_s1 = self.mri_specific_codebook(mri_spe)
        outputs = {
            "mri_com": mri_com,
            "mri_spe": mri_spe,
            "c1": c1,
            "s1": s1,
            "codebook_loss": cb_c1 + cb_s1,
            "indices": {"mri_common": idx_c1, "mri_specific": idx_s1},
        }

        if missing_pet or pet is None:
            q_common = self.mri_to_pet_common(mri_com)
            q_specific = self.mri_to_pet_specific(mri_spe)
            c2, idx_c2 = self.pet_common_codebook.nearest_from_query(q_common)
            s2, idx_s2 = self.pet_specific_codebook.nearest_from_query(q_specific)
            outputs.update({"pet_com": q_common, "pet_spe": q_specific, "c2": c2, "s2": s2})
            outputs["indices"].update({"pet_common": idx_c2, "pet_specific": idx_s2})
        else:
            pet_com, pet_spe = self.pet_encoder(pet)
            c2, cb_c2, idx_c2 = self.pet_common_codebook(pet_com)
            s2, cb_s2, idx_s2 = self.pet_specific_codebook(pet_spe)
            outputs.update({
                "pet_com": pet_com,
                "pet_spe": pet_spe,
                "c2": c2,
                "s2": s2,
                "codebook_loss": outputs["codebook_loss"] + cb_c2 + cb_s2,
            })
            outputs["indices"].update({"pet_common": idx_c2, "pet_specific": idx_s2})

        fused = self._fuse(outputs["c1"], outputs["s1"], outputs["c2"], outputs["s2"])
        logits = self.classifier(fused)
        outputs.update({"fused": fused, "logits": logits})
        if debug_shapes:
            print("mri:", mri.shape, "pet:", None if pet is None else pet.shape, "missing_pet:", missing_pet)
            print("mri_com/spe:", mri_com.shape, mri_spe.shape)
            print("c1/s1/c2/s2:", outputs["c1"].shape, outputs["s1"].shape, outputs["c2"].shape, outputs["s2"].shape)
            print("fused:", fused.shape, "logits:", logits.shape)
        return outputs


def disentangle_loss(out):
    common = F.mse_loss(out["mri_com"], out["pet_com"])
    specific_distance = torch.norm(out["mri_spe"] - out["pet_spe"], p=2, dim=1).mean().clamp_min(1e-4)
    return common / specific_distance


def reconstruction_loss(model, out):
    mri_self = model.mri_decoder(out["mri_com"], out["mri_spe"])
    pet_self = model.pet_decoder(out["pet_com"], out["pet_spe"])
    mri_cross = model.mri_decoder(out["pet_com"], out["mri_spe"])
    pet_cross = model.pet_decoder(out["mri_com"], out["pet_spe"])
    return (
        F.mse_loss(mri_self, out["mri_com"].detach())
        + F.mse_loss(pet_self, out["pet_com"].detach())
        + F.mse_loss(mri_cross, out["mri_com"].detach())
        + F.mse_loss(pet_cross, out["pet_com"].detach())
    )


def feature_match_loss(model, out):
    f1 = model.mri_to_pet_common(out["mri_com"])
    f2 = model.mri_to_pet_specific(out["mri_spe"])
    c2 = out["c2"].detach()
    s2 = out["s2"].detach()
    b = f1.size(0)
    f1_pairs = f1[:, None, :].expand(b, b, -1).reshape(b * b, -1)
    f2_pairs = f2[:, None, :].expand(b, b, -1).reshape(b * b, -1)
    c2_pairs = c2[None, :, :].expand(b, b, -1).reshape(b * b, -1)
    s2_pairs = s2[None, :, :].expand(b, b, -1).reshape(b * b, -1)
    labels = torch.eye(b, device=f1.device).reshape(-1)
    logits_common = model.common_matcher(f1_pairs, c2_pairs)
    logits_specific = model.specific_matcher(f2_pairs, s2_pairs)
    return F.binary_cross_entropy_with_logits(logits_common, labels) + F.binary_cross_entropy_with_logits(logits_specific, labels)

