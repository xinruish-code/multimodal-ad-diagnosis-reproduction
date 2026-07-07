import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder3D(nn.Module):
    def __init__(self, in_channels=1, channels=(16, 16, 32, 32, 64, 96, 80, 64), latent_dim=128):
        super().__init__()
        layers = []
        c_in = in_channels
        for idx, c_out in enumerate(channels):
            layers.extend([
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            ])
            if idx in {1, 3, 5, 7}:
                layers.append(nn.MaxPool3d(2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, x):
        fmap = self.features(x)
        pooled = F.adaptive_avg_pool3d(fmap, 1).flatten(1)
        latent = self.proj(pooled)
        return latent, fmap


class Disentangler(nn.Module):
    def __init__(self, latent_dim=128, feature_dim=64):
        super().__init__()
        self.class_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, feature_dim))
        self.meta_head = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, feature_dim))

    def forward(self, latent):
        return self.class_head(latent), self.meta_head(latent)


class Classifier(nn.Module):
    def __init__(self, feature_dim=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, num_classes))

    def forward(self, fc):
        return self.net(fc)


class MetaGenerator(nn.Module):
    def __init__(self, meta_dim, noise_dim=16, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim + noise_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
        )

    def forward(self, meta, noise):
        return self.net(torch.cat([meta, noise], dim=1))


class MetaDiscriminator(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 64), nn.LeakyReLU(0.2, inplace=True), nn.Linear(64, 1))

    def forward(self, fm):
        return self.net(fm)


class MetaQ(nn.Module):
    def __init__(self, feature_dim=64, meta_dim=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, meta_dim))

    def forward(self, fm):
        return self.net(fm)


class Reconstructor(nn.Module):
    def __init__(self, feature_dim=64, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )

    def forward(self, fc, fm):
        return self.net(torch.cat([fc, fm], dim=1))


class CLUB(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, feature_dim))
        self.logvar = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, feature_dim), nn.Tanh())

    def forward(self, fc, fm):
        mu = self.mu(fm)
        logvar = self.logvar(fm)
        positive = -((fc - mu) ** 2) / (2.0 * logvar.exp())
        random_index = torch.randperm(fc.size(0), device=fc.device)
        negative = -((fc[random_index] - mu) ** 2) / (2.0 * logvar.exp())
        return (positive.sum(dim=1) - negative.sum(dim=1)).mean()


class MMRN(nn.Module):
    def __init__(self, meta_dim, num_classes=2, latent_dim=128, feature_dim=64, noise_dim=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.encoder = ConvEncoder3D(latent_dim=latent_dim)
        self.disentangler = Disentangler(latent_dim, feature_dim)
        self.classifier = Classifier(feature_dim, num_classes)
        self.generator = MetaGenerator(meta_dim, noise_dim, feature_dim)
        self.discriminator = MetaDiscriminator(feature_dim)
        self.q_net = MetaQ(feature_dim, meta_dim)
        self.reconstructor = Reconstructor(feature_dim, latent_dim)
        self.club = CLUB(feature_dim)

    def encode_view(self, x):
        latent, fmap = self.encoder(x)
        fc, fm = self.disentangler(latent)
        logits = self.classifier(fc)
        return {"latent": latent, "fmap": fmap, "fc": fc, "fm": fm, "logits": logits}

    def forward(self, view_i, view_j=None, meta=None, debug_shapes=False):
        oi = self.encode_view(view_i)
        oj = self.encode_view(view_j) if view_j is not None else None
        out = {"i": oi, "j": oj}
        if meta is not None:
            noise = torch.randn(meta.size(0), self.noise_dim, device=meta.device)
            out["generated_fm"] = self.generator(meta, noise)
        if debug_shapes:
            print("view_i:", view_i.shape)
            if view_j is not None:
                print("view_j:", view_j.shape)
            print("latent_i:", oi["latent"].shape, "fmap_i:", oi["fmap"].shape)
            print("fc_i:", oi["fc"].shape, "fm_i:", oi["fm"].shape)
            if oj is not None:
                print("fc_j:", oj["fc"].shape, "fm_j:", oj["fm"].shape)
            print("logits_i:", oi["logits"].shape)
            if "generated_fm" in out:
                print("generated_fm:", out["generated_fm"].shape)
        return out
