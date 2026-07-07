# AAGN Notes

## Paper

**Title:** Anatomy-Aware Gating Network for Explainable Alzheimer's Disease Diagnosis

## Modality

- Structural MRI

## Main Task

The model is designed for explainable Alzheimer's disease diagnosis using structural MRI data.

Main classification tasks in my reproduction include:

- AD vs CN
- AD vs MCI
- MCI vs CN

## Core Idea

AAGN focuses on anatomy-aware and explainable Alzheimer's disease diagnosis.

Many CNN-based medical imaging models can make predictions, but it is often difficult to understand which brain regions are important for the decision.

AAGN tries to solve this problem by explicitly using anatomical brain regions during feature learning. Instead of only using post-hoc explanation methods, the model directly selects important brain regions inside the network.

In simple terms, AAGN asks:

1. Which brain regions are useful for AD diagnosis?
2. How can the model use anatomical knowledge during prediction?
3. Can the model be more explainable by selecting disease-related ROIs?

## Main Components

### 1. 3D CNN Backbone

AAGN uses a 3D CNN to extract volumetric features from structural MRI.

Because MRI is a 3D brain image, 3D convolution is used to capture spatial patterns across the whole brain.

### 2. Brain Atlas-Based ROI Definition

The model uses an anatomical brain atlas to define regions of interest, also called ROIs.

Each ROI corresponds to a specific anatomical brain region.

This allows the model to organize features according to brain anatomy instead of treating the image as only raw voxels.

### 3. Anatomy-Aware Squeeze-and-Excite Module

The anatomy-aware squeeze-and-excite module extracts ROI-level information from CNN feature maps.

It learns which feature channels are important for each anatomical region.

This helps the model connect deep features with meaningful brain regions.

### 4. Anatomy Gating Module

The anatomy gating module selects important ROIs for diagnosis.

It uses a differentiable gating strategy so that the model can learn ROI selection during training.

### 5. Gumbel-Softmax

Gumbel-Softmax is used to make the ROI selection process trainable.

It allows the model to perform a selection-like operation while still supporting gradient-based optimization.

### 6. Explainability

AAGN provides built-in ROI-level interpretability.

The selected ROIs can be analyzed to understand which anatomical regions contribute more to the prediction.

This is especially useful for medical AI because doctors and researchers care not only about the prediction result, but also about the reasoning behind it.

## Reproduction Setting

In my reproduction, the model was adapted to pairwise AD classification tasks using ADNI structural MRI data.

| Task | Setting |
|---|---|
| AD vs CN | Binary classification |
| AD vs MCI | Binary classification |
| MCI vs CN | Binary classification |

## Reproduced Results

| Task | ACC (%) | SEN (%) | SPE (%) | AUC (%) | F1 (%) | PRE (%) |
|---|---:|---:|---:|---:|---:|---:|
| AD vs CN | 87.1 | 74.5 | 92.7 | 93.4 | 78.1 | 82.2 |
| AD vs MCI | 76.0 | 49.8 | 89.7 | 81.5 | 58.8 | 72.7 |
| MCI vs CN | 63.5 | 58.8 | 67.5 | 68.2 | 59.3 | 64.2 |

## Observations

1. AAGN achieves strong performance on AD vs CN using only structural MRI.
2. The model is especially valuable because it provides ROI-level interpretability.
3. AD vs MCI and MCI vs CN remain more difficult than AD vs CN.
4. The anatomy-aware design makes the model more clinically meaningful than a standard black-box CNN.
5. The gating mechanism can help identify potentially disease-related anatomical regions.

## Reproduction Challenges

- ROI mask preparation is important and can affect the final result.
- Atlas registration must be consistent with MRI preprocessing.
- Gumbel-Softmax and gating modules require careful implementation.
- The selected ROIs need to be interpreted carefully.
- Training 3D CNN models can be computationally expensive.

## Status

Reproduced.
