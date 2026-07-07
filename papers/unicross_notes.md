# UniCross Notes

## Paper

**Title:** Balanced Multimodal Learning for Alzheimer's Disease Diagnosis by Uni-modal Separation and Metadata-guided Cross-modal Interaction

## Modality

- Structural MRI
- FDG-PET
- Metadata

## Main Task

The model is designed for Alzheimer's disease diagnosis using multimodal neuroimaging data and metadata.

The main classification settings may include:

- AD vs CN
- AD vs MCI
- MCI vs CN
- CN vs MCI vs AD
- MCI conversion prediction

## Core Idea

UniCross focuses on balanced multimodal learning for Alzheimer's disease diagnosis.

In multimodal medical AI, one modality may dominate the learning process. For example, PET may contain stronger disease-related signals than MRI, so the model may over-rely on PET and ignore useful MRI information.

This problem is called modality laziness.

UniCross tries to solve this issue by separating unimodal learning and then guiding cross-modal interaction using metadata.

In simple terms, UniCross asks:

1. How can MRI and PET each learn useful disease-related features?
2. How can the model avoid relying only on the stronger modality?
3. How can metadata help guide the interaction between modalities?
4. How can multimodal fusion become more balanced and clinically meaningful?

## Main Components

### 1. MRI Encoder

The MRI encoder extracts structural brain features from MRI images.

MRI mainly captures anatomical changes such as brain atrophy and structural abnormalities.

### 2. PET Encoder

The PET encoder extracts metabolic features from FDG-PET images.

PET can provide complementary information related to glucose metabolism and disease progression.

### 3. Uni-modal Separation

UniCross first encourages each modality to learn useful representations independently.

This helps prevent one modality from being ignored during training.

For example, the MRI branch should still be able to learn disease-related features even when PET is available.

### 4. MRI-specific Classifier

The MRI-specific classifier predicts disease labels using only MRI features.

This forces the MRI encoder to learn discriminative information.

### 5. PET-specific Classifier

The PET-specific classifier predicts disease labels using only PET features.

This ensures that the PET encoder also learns meaningful disease-related features.

### 6. Shared Classifier Head

The shared classifier head encourages MRI and PET features to be organized in a comparable feature space.

This helps prepare the two modalities for later fusion.

### 7. Metadata-guided Cross-modal Interaction

UniCross uses metadata such as age, gender, and education to guide the interaction between MRI and PET features.

The idea is that patients with similar disease status and similar metadata may have more comparable multimodal patterns.

### 8. Metadata-weighted Contrastive Loss

The metadata-weighted contrastive loss encourages meaningful relationships in the feature space.

Samples with similar labels and similar metadata are pulled closer together.

Samples with different labels are pushed apart.

This helps the model learn more clinically structured representations.

### 9. Two-stage Training Strategy

UniCross uses a two-stage training strategy.

In the first stage, the model trains unimodal encoders and encourages each modality to learn useful features.

In the second stage, the encoders are frozen or partially frozen, and the fusion module is trained for final multimodal diagnosis.

This design helps reduce modality laziness.

## Reproduction Setting

In my reproduction or adaptation, UniCross can be organized around multimodal MRI-PET classification tasks using ADNI data.

Possible classification settings include:

| Task | Setting |
|---|---|
| AD vs CN | Binary classification |
| AD vs MCI | Binary classification |
| MCI vs CN | Binary classification |
| CN vs MCI vs AD | Multiclass classification |
| MCI conversion | Conversion prediction |

## Expected Advantages

1. UniCross directly addresses modality laziness.
2. It encourages both MRI and PET to learn useful disease-related features.
3. Metadata provides additional clinical guidance for multimodal representation learning.
4. The two-stage training strategy makes the fusion process more balanced.
5. The model is suitable for studying modern multimodal AD diagnosis.

## Reproduction Challenges

- MRI and PET preprocessing must be carefully aligned.
- Metadata need to be cleaned and matched with imaging data.
- Missing metadata or missing modalities may affect training.
- Contrastive learning requires careful batch construction.
- The two-stage training process is more complicated than standard end-to-end training.
- Hyperparameters for the metadata-weighted contrastive loss may strongly affect performance.

## Difference from Previous Models

Compared with the previous reproduced models:

- UniCross focuses more explicitly on balanced multimodal learning.
- It addresses modality laziness, which is important in MRI-PET fusion.
- It uses metadata to guide cross-modal interaction.
- It is more recent and more aligned with current multimodal medical AI research.
- It is a strong candidate for extending the original reproduction project.

## Current Status

Reproduced / adapting.
