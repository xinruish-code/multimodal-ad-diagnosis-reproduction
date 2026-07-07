# MMRN Notes

## Paper

**Title:** Multi-Template Meta-Information Regularized Network for Alzheimer's Disease Diagnosis Using Structural MRI

## Modality

- Structural MRI

## Main Task

The model is designed for Alzheimer's disease diagnosis using structural MRI data.

Main classification tasks in my reproduction include:

- AD vs CN
- AD vs MCI
- MCI vs CN

## Core Idea

MMRN focuses on improving Alzheimer's disease diagnosis by reducing the influence of non-disease-related meta-information.

In medical imaging data, factors such as age, gender, and education may affect brain structure. These factors can become confounders if the model learns them instead of true disease-related patterns.

MMRN tries to separate disease-related features from meta-information-related features.

## Main Components

### 1. Multi-Template Input

The model uses MRI images aligned to multiple templates.

Different templates may capture different anatomical representations, which can help the model learn more robust brain features.

### 2. Siamese Encoder

MMRN uses encoders with shared weights to process different template-based MRI inputs.

The shared encoder design encourages the model to learn consistent representations across templates.

### 3. Self-Supervised Learning

The model uses self-supervised learning to help distinguish common disease-related features from template-specific information.

### 4. Meta-Information Regularization

The model considers meta-information such as:

- age
- gender
- education

The goal is to prevent the model from overusing these confounding factors.

### 5. Mutual Information Minimization

MMRN uses mutual information minimization to reduce dependency between disease-related features and meta-information-related features.

This encourages the learned disease representation to be more independent and clinically meaningful.

## Reproduction Setting

In my reproduction, the model was adapted to pairwise AD classification tasks using ADNI structural MRI data.

| Task | Setting |
|---|---|
| AD vs CN | Binary classification |
| AD vs MCI | Binary classification |
| MCI vs CN | Binary classification |

## Reproduced Results

| Task | ACC (%) | SEN (%) | SPE (%) | AUC (%) |
|---|---:|---:|---:|---:|
| AD vs CN | 88.5 | 73.6 | 95.1 | 94.4 |
| AD vs MCI | 78.6 | 52.8 | 92.0 | 81.9 |
| MCI vs CN | 59.0 | 53.6 | 63.6 | 63.6 |

## Observations

1. MMRN performs well on AD vs CN, showing that structural MRI contains useful disease-related information.
2. AD vs MCI is more difficult because MCI subjects may share features with both AD and CN.
3. MCI vs CN has weaker performance, suggesting that subtle MCI-stage changes are harder to detect using only structural MRI.
4. Meta-information regularization is useful because it makes the model less dependent on confounding variables.

## Reproduction Challenges

- Multi-template preprocessing can be complicated and time-consuming.
- The model requires careful organization of MRI images from different templates.
- Mutual information minimization introduces additional training complexity.
- If metadata are incomplete or noisy, the regularization module may be affected.
- The reproduced setting may differ from the original paper due to dataset split and preprocessing differences.

## Status

Reproduced.
