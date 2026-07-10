# MOGAD-Net Notes

## Paper

**Title:** Multi-organ guided diagnosis of mild cognitive impairment via hierarchical alignment and knowledge distillation

## Modality

- Total-body FDG-PET
- Brain FDG-PET
- Heart FDG-PET
- Gut FDG-PET
- Brain T1 MRI

## Main Task

The model is designed for mild cognitive impairment diagnosis.

The main classification task in my reproduction is:

- MCI vs NC

Unlike many previous Alzheimer's disease diagnosis models that focus mainly on brain-only MRI or PET images, MOGAD-Net introduces a multi-organ perspective. It uses systemic pathological information from the brain, heart, and gut during training, while allowing inference using only brain images.

## Core Idea

MOGAD-Net is based on the idea that mild cognitive impairment is not only a brain-related condition, but may also involve systemic pathological and metabolic changes in peripheral organs.

The model uses total-body FDG-PET images to extract features from:

- brain
- heart
- gut

During training, the multi-organ model learns richer systemic diagnostic information. Then, through hierarchical knowledge distillation, the diagnostic knowledge from the multi-organ teacher model is transferred to brain-only student models.

In simple terms, MOGAD-Net tries to answer three questions:

1. Can heart and gut information help MCI diagnosis?
2. Can multi-organ knowledge improve a brain-only diagnostic model?
3. Can the final model remain clinically practical by requiring only brain FDG-PET or brain T1 MRI during inference?

## Main Components

### 1. PanSwin Encoder

MOGAD-Net introduces a panoramic Swin Transformer encoder, called PanSwin, as the main feature extractor.

PanSwin is designed to capture global features and long-range dependencies from 3D medical images. This is important because MCI-related pathological changes may be subtle and spatially distributed.

Compared with standard Swin Transformer, PanSwin introduces global modeling through Global Swin Transformer Blocks.

### 2. Global Swin Transformer Block

The Global Swin Transformer Block is designed to overcome the limited global modeling ability of standard window-based Swin Transformer blocks.

It combines:

- local feature extraction from Swin Transformer blocks
- global information compression from the TGIC module

This allows the model to preserve both local and global pathological information.

### 3. Token-based Global Information Compression

The Token-based Global Information Compression module compresses high-dimensional feature maps while preserving global contextual information.

This helps reduce computational cost while still allowing the network to capture long-range relationships across anatomical regions.

### 4. Pretraining Brain FDG-PET Network

The first step is to train a brain FDG-PET model for MCI vs NC classification.

This pretrained brain branch is later used to generate reliable pseudo-labels for heart and gut images in the semi-supervised multi-organ learning phase.

### 5. Semi-supervised Multi-organ Collaboration

In Phase 1, MOGAD-Net uses three organ-specific branches:

- brain FDG-PET branch
- heart FDG-PET branch
- gut FDG-PET branch

The brain branch is initialized from the pretrained brain FDG-PET model and kept frozen. The heart and gut branches are trainable.

For unlabeled total-body FDG-PET data, high-confidence predictions from the frozen brain branch are used as pseudo-labels to guide the heart and gut branches.

### 6. Hierarchical Feature Alignment

The Hierarchical Feature Alignment module aligns heart and gut features with brain features at multiple network layers.

Instead of only aligning final-layer features, the model aligns features hierarchically across different depths. This encourages heart and gut representations to become more diagnosis-relevant and more consistent with brain representations.

### 7. Label Consistency Loss

The label consistency loss encourages multi-organ features from subjects with the same label to be closer in feature space, while pushing features from subjects with different labels farther apart.

This improves discriminative representation learning for MCI vs NC classification.

### 8. Compression Fusion Module

The Compression Fusion Module integrates features from brain, heart, and gut FDG-PET images.

It compresses organ-specific features and then uses attention-based fusion to generate a unified multi-organ representation.

### 9. Hierarchical-constraint Knowledge Distillation

In Phase 2, the trained multi-organ model serves as a teacher.

The student models use only:

- brain FDG-PET
- or brain T1 MRI

The student models learn not only from classification labels, but also from multi-scale feature representations of the multi-organ teacher.

This allows the final brain-only model to benefit from multi-organ information during training, while remaining clinically practical during inference.

## Reproduction Setting

In my reproduction, the model was adapted to MCI vs NC classification.

The reproduced setting can be organized as follows:

| Phase | Training Input | Testing Input | Purpose |
|---|---|---|---|
| Pretraining | Brain FDG-PET | Brain FDG-PET | Train a brain-based baseline model |
| Phase 1 | Brain-heart-gut FDG-PET | Brain-heart-gut FDG-PET | Train a multi-organ teacher model |
| Phase 2 | Multi-organ teacher + brain-only input | Brain FDG-PET or brain T1 MRI | Distill multi-organ knowledge into brain-only models |
| Inference | Brain FDG-PET or brain T1 MRI | Brain FDG-PET or brain T1 MRI | Clinically practical MCI diagnosis |

## Reproduced Results

The reproduced results are based on my own implementation and experiment logs.

| Model / Phase | Training Input | Testing Input | AUC (%) | ACC (%) | SEN (%) | SPE (%) | F1 (%) |
|---|---|---|---:|---:|---:|---:|---:|
| Pretraining | Brain FDG-PET | Brain FDG-PET | TODO | TODO | TODO | TODO | TODO |
| Phase 1 | Brain-heart-gut FDG-PET | Brain-heart-gut FDG-PET | TODO | TODO | TODO | TODO | TODO |
| Phase 2 | BHG FDG-PET | Brain FDG-PET | TODO | TODO | TODO | TODO | TODO |
| Phase 2 | BHG FDG-PET + Brain T1 | Brain T1 MRI | TODO | TODO | TODO | TODO | TODO |

## Observations

1. MOGAD-Net introduces a systemic multi-organ perspective for MCI diagnosis.
2. The model uses brain, heart, and gut FDG-PET information during training, but only requires brain images during inference.
3. The semi-supervised learning design reduces reliance on fully labeled total-body PET data.
4. Hierarchical feature alignment encourages heart and gut features to align with brain diagnostic representations.
5. Hierarchical knowledge distillation improves clinical practicality by transferring multi-organ knowledge into brain-only models.
6. PanSwin improves global feature extraction compared with standard CNN or window-limited Swin Transformer backbones.
7. This model is useful as a modern example of training-time multi-organ guidance with test-time single-organ inference.

## Reproduction Challenges

- Total-body FDG-PET data are difficult to obtain.
- Brain, heart, and gut regions require separate preprocessing pipelines.
- Heart and gut segmentation may introduce additional preprocessing errors.
- Semi-supervised pseudo-label quality can affect heart and gut branch training.
- Knowledge distillation requires careful alignment between teacher and student features.
- The full pipeline has multiple phases, making reproduction more complex than standard end-to-end classification models.
- Brain-only inference requires careful validation to show that multi-organ knowledge has been successfully transferred.
- Results should be verified from training scripts, slurm files, logs, and output folders rather than copied directly from the paper.

## Difference from Previous Models

Compared with the previous reproduced models:

- MOGAD-Net focuses on MCI vs NC rather than AD/MCI/CN pairwise classification.
- It uses multi-organ FDG-PET information instead of only brain MRI or brain PET.
- It uses semi-supervised learning to exploit unlabeled total-body PET data.
- It transfers multi-organ knowledge into brain-only models through hierarchical knowledge distillation.
- It is designed for clinical practicality because inference only requires brain FDG-PET or brain T1 MRI.

## Status

Reproduced.
