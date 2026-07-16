# DCFMnet Notes

## Paper

**Title:** Disentanglement and codebook learning-induced feature match network to diagnose neurodegenerative diseases on incomplete multimodal data

**Short name:** DCFMnet

**Venue:** Pattern Recognition, 2025

**Code / dataset link:** https://github.com/Meiyan88/DCFMnet

## Modality

The paper studies incomplete multimodal neuroimaging data.

Main modality settings include:

- ADNI: MRI + PET
- PPMI: MRI + DTI / FA

In the ADNI setting, all subjects have MRI at baseline, while only part of the subjects have PET. This creates the missing-modality problem.

## Main Task

DCFMnet is designed for neurodegenerative disease diagnosis with incomplete multimodal data.

Main tasks include:

- AD vs NC
- pMCI vs sMCI
- PD vs NC
- PD vs SWEDD

For this repository, the most relevant part is the ADNI setting:

```text
MRI + PET incomplete multimodal diagnosis for Alzheimer's disease
```

## Core Idea

DCFMnet does not use a teacher-student distillation framework.

Instead, it handles missing modalities through:

1. feature disentanglement;
2. modality-common and modality-specific codebooks;
3. feature matching for missing latent modal features;
4. disease diagnosis based on fused latent features.

In simple terms:

> DCFMnet does not generate the missing PET image. It uses MRI to retrieve the most compatible missing PET latent feature from learned PET codebooks.

## Pipeline

```text
Input multimodal data
MRI + PET

↓
Feature Disentanglement Module
Each modality is split into:
- modality-common feature
- modality-specific feature

↓
Feature Match Module
Each common/specific feature is matched to a learned codebook.
If one modality is missing, its latent feature is retrieved from the corresponding codebook using available MRI features.

↓
Disease Diagnosis Module
Latent modality-common and modality-specific features are fused.

↓
Diagnosis
AD vs NC / pMCI vs sMCI / PD vs NC / PD vs SWEDD
```

## Main Components

### 1. Feature Disentanglement Module

Each modality is decomposed into two parts:

- **Modality-common feature:** shared disease information across modalities.
- **Modality-specific feature:** unique complementary information from each modality.

For example, in MRI+PET AD diagnosis:

- MRI-specific information may reflect structural atrophy.
- PET-specific information may reflect metabolic abnormality.
- Common information reflects disease-related patterns shared by both.

This module is important because directly forcing different modalities into one common space may lose modality-specific diagnostic information.

### 2. Reconstruction Constraints

DCFMnet uses self-reconstruction and cross-reconstruction losses.

The goal is to ensure that the disentangled common and specific features still preserve meaningful information.

In simple terms:

```text
The model can split features,
but after splitting, the information should still be useful enough to reconstruct the modality.
```

### 3. Feature Codebooks

A codebook is a learnable library of typical latent feature patterns.

DCFMnet builds separate codebooks for each modality and each feature type:

- MRI-common codebook
- MRI-specific codebook
- PET-common codebook
- PET-specific codebook

The model maps continuous extracted features to their nearest learned latent prototypes in the codebooks.

### 4. Feature Match Module

This module is the key to missing-modality handling.

If PET is missing at testing time, DCFMnet does not synthesize a PET image.

Instead, it uses MRI-derived additional features to search the PET codebooks and retrieve:

- missing PET-common latent feature
- missing PET-specific latent feature

Then the model fuses:

```text
MRI-common latent feature
MRI-specific latent feature
retrieved PET-common latent feature
retrieved PET-specific latent feature
```

for classification.

### 5. Disease Diagnosis Module

The final feature fusion follows this idea:

- modality-common latent features are combined through weighted summation;
- modality-specific latent features are concatenated;
- the fused representation is passed to a classifier.

## Why It Is a Missing-Modality Method

DCFMnet is a missing-modality method because it can use incomplete multimodal data during training and can perform diagnosis when one modality is unavailable at testing time.

```text
Training: complete and incomplete multimodal data
Testing: MRI only or complete MRI+PET
```

It solves missing modality at the latent-feature level rather than at the image-generation level.

## Reproduction Setting

| Item | Setting |
|---|---|
| Main ADNI modalities | MRI + PET |
| Missing modality | PET may be unavailable |
| Teacher-student? | No |
| Core modules | Feature disentanglement + codebook learning + feature matching |
| Main ADNI tasks | AD vs NC, pMCI vs sMCI |
| Metrics to report | ACC, SEN, SPE, F1, AUC |

## Result Reporting Rule

Because DCFMnet is not a teacher-student distillation method, the result table should not force teacher/student roles.

Instead, report complete-modality and missing-modality settings separately.

Use mean ± standard deviation when fold-level or repeated-run results are available.

| Task | Setting | Method | Training Input | Inference Input | ACC | SEN | SPE | F1 | AUC |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| AD vs NC | Complete modality | DCFMnet-C | incomplete multimodal training | MRI+PET | TODO | TODO | TODO | TODO | TODO |
| AD vs NC | Missing modality | DCFMnet-M | incomplete multimodal training | MRI only | TODO | TODO | TODO | TODO | TODO |
| pMCI vs sMCI | Complete modality | DCFMnet-C | incomplete multimodal training | MRI+PET | TODO | TODO | TODO | TODO | TODO |
| pMCI vs sMCI | Missing modality | DCFMnet-M | incomplete multimodal training | MRI only | TODO | TODO | TODO | TODO | TODO |

## Key Strengths

1. Does not require image-level PET imputation.
2. Preserves both modality-common and modality-specific information.
3. Uses learned codebooks to retrieve missing latent modal features.
4. Can use incomplete multimodal data instead of discarding subjects with missing modalities.
5. Supports both AD-related and PD-related neurodegenerative disease diagnosis tasks.

## Limitations and Reproduction Challenges

- Codebook learning and nearest-neighbor lookup can be sensitive to implementation details.
- Missing modality retrieval depends on whether MRI-PET latent correspondence is learned well.
- Codebook features are less directly interpretable than ROI-level anatomical attention.
- This method is not a teacher-student framework, so teacher/student reporting is not applicable.
- Per-fold metrics should still be saved to compute mean ± standard deviation.

## Status

Paper notes prepared. Reproduction not yet completed.
