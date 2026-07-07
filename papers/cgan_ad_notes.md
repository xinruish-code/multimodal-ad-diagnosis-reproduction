# CGAN-AD Notes

## Paper

**Title:** A coupled-GAN architecture to fuse MRI and PET image features for multi-stage classification of Alzheimer's disease

## Modality

- MRI
- PET

## Main Task

The model is designed for Alzheimer's disease stage classification using multimodal MRI and PET data.

The main classification setting includes multiple Alzheimer's disease stages, such as:

- CN
- MCI
- AD

## Core Idea

CGAN-AD uses a coupled generative adversarial network to fuse MRI and PET image features.

MRI and PET provide complementary information:

- MRI captures structural brain atrophy and anatomical changes.
- PET captures metabolic or functional abnormalities related to Alzheimer's disease.

The key idea of CGAN-AD is not simply to concatenate MRI and PET features. Instead, it learns a shared latent representation between the two modalities.

In simple terms, the model tries to answer:

1. How can MRI and PET be mapped into a common feature space?
2. How can the model preserve useful information from both modalities?
3. How can fused MRI-PET features improve AD stage classification?

## Main Components

### 1. MRI Encoder

The MRI encoder extracts features from MRI images.

It learns structural brain information such as atrophy patterns and anatomical abnormalities.

### 2. PET Encoder

The PET encoder extracts features from PET images.

It learns metabolic or functional patterns that may be related to Alzheimer's disease progression.

### 3. Shared Latent Space

The model maps MRI and PET features into a shared latent space.

This shared space is designed to contain useful information from both modalities.

The goal is to make the MRI and PET representations compatible and complementary.

### 4. Dual Decoders

CGAN-AD uses decoders to reconstruct modality-specific information from the shared latent representation.

This reconstruction process helps ensure that the shared representation still preserves important MRI and PET information.

### 5. Dual Discriminators

The model uses adversarial learning through discriminators.

The discriminators encourage the generated or reconstructed representations to be more realistic and modality-consistent.

### 6. Feature Fusion

After learning MRI and PET representations, the model fuses the features for final disease classification.

The fusion strategy aims to combine structural and metabolic information more effectively than simple feature concatenation.

### 7. Classification Head

The fused feature representation is passed into a classifier to predict the disease stage.

The final output can be used for AD-related classification tasks.

## Reproduction Setting

In my reproduction or adaptation, the model can be organized around multimodal MRI-PET classification tasks.

Possible classification settings include:

| Task | Setting |
|---|---|
| AD vs CN | Binary classification |
| AD vs MCI | Binary classification |
| MCI vs CN | Binary classification |
| CN vs MCI vs AD | Multiclass classification |

## Expected Advantages

1. CGAN-AD can use both MRI and PET information.
2. The shared latent space helps align two different imaging modalities.
3. Reconstruction learning helps preserve modality-specific information.
4. Adversarial learning may improve the quality of fused representations.
5. The model is suitable for studying multimodal fusion in Alzheimer's disease diagnosis.

## Reproduction Challenges

- MRI and PET need careful preprocessing and spatial alignment.
- Subjects with missing modalities may be difficult to include.
- GAN-based training can be unstable.
- The balance between reconstruction loss, adversarial loss, and classification loss needs careful tuning.
- Multiclass classification is more difficult than binary classification.
- The original implementation may need adaptation for available ADNI data splits.

## Difference from Previous Models

Compared with the previous five reproduced models:

- CGAN-AD is more explicitly focused on multimodal MRI-PET fusion.
- It uses generative adversarial learning rather than only CNN or graph-based feature extraction.
- It tries to learn a shared latent representation between MRI and PET.
- It is useful for studying how two imaging modalities can complement each other.

## Current Status

Reproduced / adapting.
