import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Encoder3D(nn.Module):
    def __init__(self, latent_channels=128, base=16):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock3D(1, base, stride=2),
            ConvBlock3D(base, base * 2, stride=2),
            ConvBlock3D(base * 2, base * 4, stride=2),
            ConvBlock3D(base * 4, latent_channels, stride=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder3D(nn.Module):
    def __init__(self, latent_channels=128, base=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            ConvBlock3D(latent_channels, base * 4),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            ConvBlock3D(base * 4, base * 2),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            ConvBlock3D(base * 2, base),
            nn.Conv3d(base, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator3D(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock3D(1, base, stride=1),
            ConvBlock3D(base, base, stride=2),
            ConvBlock3D(base, base * 2, stride=1),
            ConvBlock3D(base * 2, base * 2, stride=2),
            ConvBlock3D(base * 2, base * 4, stride=1),
            ConvBlock3D(base * 4, base * 4, stride=2),
        )
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x):
        feat = self.features(x).mean(dim=(2, 3, 4))
        return self.head(feat).squeeze(-1)


class CGANC(nn.Module):
    def __init__(self, latent_channels=128, base=16):
        super().__init__()
        self.mri_encoder = Encoder3D(latent_channels, base)
        self.pet_encoder = Encoder3D(latent_channels, base)
        self.mri_decoder = Decoder3D(latent_channels, base)
        self.pet_decoder = Decoder3D(latent_channels, base)

    def encode(self, mri, pet):
        z_mri = self.mri_encoder(mri)
        z_pet = self.pet_encoder(pet)
        fused = z_mri + z_pet
        return fused, z_mri, z_pet

    def forward(self, mri, pet, debug_shapes=False):
        fused, z_mri, z_pet = self.encode(mri, pet)
        rec_mri = self.mri_decoder(fused)
        rec_pet = self.pet_decoder(fused)
        if debug_shapes:
            print("mri:", mri.shape, "pet:", pet.shape)
            print("z_mri:", z_mri.shape, "z_pet:", z_pet.shape, "fused:", fused.shape)
            print("rec_mri:", rec_mri.shape, "rec_pet:", rec_pet.shape)
        return {
            "fused": fused,
            "z_mri": z_mri,
            "z_pet": z_pet,
            "rec_mri": rec_mri,
            "rec_pet": rec_pet,
        }


class LatentClassifier(nn.Module):
    def __init__(self, latent_channels=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(latent_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, fused, debug_shapes=False):
        feat_map = self.net(fused)
        feat = feat_map.mean(dim=(2, 3, 4))
        logits = self.head(feat)
        if debug_shapes:
            print("classifier_input:", fused.shape)
            print("classifier_feat_map:", feat_map.shape, "feat:", feat.shape, "logits:", logits.shape)
        return logits, feat
