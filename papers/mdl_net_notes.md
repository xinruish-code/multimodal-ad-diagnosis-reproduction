# MDL-Net Notes

## Paper

**Title:** 3D Multimodal Fusion Network With Disease-Induced Joint Learning for Early Alzheimer's Disease Diagnosis

## Modality

- Structural MRI
- PET

## Main Task

The model is designed for Alzheimer's disease diagnosis using multimodal neuroimaging data.

Main classification tasks in my reproduction include:

- AD vs CN
- AD vs MCI
- MCI vs CN

## Core Idea

MDL-Net uses both sMRI and PET to improve early Alzheimer's disease diagnosis.

The key idea is that MRI and PET provide complementary information:

- MRI captures anatomical and structural brain changes.
- PET captures metabolic or functional abnormality related to disease progression.

Instead of simply concatenating MRI and PET features, MDL-Net introduces disease-induced joint learning to guide the model toward disease-related brain regions and multimodal representations.

## Main Components

### 1. 3D ResNet Backbone

The model uses a 3D ResNet-style backbone to extract volumetric features from MRI and PET images.

Because MRI and PET are 3D medical images, 3D convolution is used to preserve spatial information across brain volumes.

### 2. Multimodal Joint Learning

MRI and PET features are learned jointly so that the model can combine structural and metabolic information.

### 3. Disease-Induced Region-Aware Learning

The model pays attention to disease-related brain regions, especially ROI-level representations based on anatomical brain regions.

This helps the model focus more on clinically meaningful areas instead of treating the whole brain equally.

### 4. Fusion Module

The extracted MRI and PET features are fused for final classification.

The fusion module is important because the model needs to combine information from two different imaging modalities.

## Reproduction Setting

In my reproduction, the model was adapted to pairwise AD classification tasks using ADNI data.

The reproduced tasks include:

| Task | Setting |
|---|---|
| AD vs CN | Binary classification |
| AD vs MCI | Binary classification |
| MCI vs CN | Binary classification |

## Reproduced Results

| Task | ACC (%) | SEN (%) | SPE (%) | AUC (%) |
|---|---:|---:|---:|---:|
| AD vs CN | 88.5 | 77.6 | 93.3 | 96.1 |
| AD vs MCI | 78.3 | 60.6 | 87.5 | 85.3 |
| MCI vs CN | 66.4 | 44.4 | 83.9 | 75.0 |

## Observations

1. AD vs CN achieves the strongest performance, especially with a high AUC.
2. MCI-related classification is more difficult because MCI is a transitional and heterogeneous stage.
3. PET provides useful complementary information to MRI.
4. The disease-region-aware design improves interpretability compared with a purely black-box CNN model.

## Reproduction Challenges

- MRI and PET preprocessing requires careful alignment.
- Some subjects may have missing modalities.
- Pairwise classification settings may differ from the original paper.
- Training 3D models is computationally expensive.
- Cross-validation can be time-consuming on limited GPU resources.

## Status

Reproduced.
