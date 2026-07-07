# LA-GMF Notes

## Paper

**Title:** Interpretable Medical Deep Framework by Logits-Constraint Attention Guiding Graph-Based Multi-Scale Fusion for Alzheimer's Disease Analysis

## Modality

- Structural MRI
- Gray matter tissue maps

## Main Task

The model is designed for Alzheimer's disease diagnosis using structural MRI data.

Main classification tasks in my reproduction include:

- AD vs CN
- AD vs MCI
- MCI vs CN

## Core Idea

LA-GMF focuses on both classification performance and interpretability.

The model uses graph-based multi-scale fusion to combine features from different levels of a 3D CNN. It also introduces logits-constrained attention to guide the model toward more meaningful disease-related regions.

In simple terms, LA-GMF tries to answer two questions at the same time:

1. Can the model classify Alzheimer's disease accurately?
2. Can the model show which brain regions or feature patterns are important for the decision?

## Main Components

### 1. 3D ResNet Backbone

LA-GMF uses a 3D ResNet-style backbone to extract volumetric brain features from gray matter maps.

Because MRI data are 3D volumes, 3D convolution helps preserve spatial information across brain regions.

### 2. Multi-Scale Feature Extraction

Different layers of the network capture different levels of information:

- shallow layers capture local and low-level patterns
- deeper layers capture more abstract disease-related patterns

LA-GMF uses multi-scale features instead of relying only on the final feature layer.

### 3. Graph-Based Multi-Scale Fusion

The model treats multi-scale feature representations as graph nodes and models their relationships.

This helps the model fuse information from different feature levels more effectively.

### 4. Logits-Constraint Attention

The attention module is constrained by classification logits.

This means the attention map is encouraged to focus on regions that are actually useful for classification, rather than producing arbitrary attention patterns.

### 5. Interpretability

The attention mechanism provides a way to understand which parts of the brain contribute more strongly to the model prediction.

This is important in medical AI because clinicians need more than just a final label.

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
| AD vs CN | 93.08 | 95.80 | 86.97 | 94.62 | 95.03 | 94.34 |
| AD vs MCI | 80.00 | 85.34 | 69.79 | 82.70 | 84.80 | 84.83 |
| MCI vs CN | 67.11 | 56.31 | 76.30 | 67.97 | 61.02 | 67.22 |

## Observations

1. LA-GMF achieves strong performance on AD vs CN.
2. Compared with some other sMRI-based models, LA-GMF shows strong sensitivity for AD vs MCI.
3. The graph-based multi-scale fusion module helps combine information from different CNN layers.
4. The logits-constrained attention mechanism improves interpretability by linking attention to classification behavior.
5. MCI vs CN remains challenging because MCI changes are subtle and heterogeneous.

## Reproduction Challenges

- Multi-scale feature extraction requires careful implementation.
- Graph-based fusion adds model complexity.
- Attention maps need to be interpreted carefully.
- Gray matter preprocessing can affect final performance.
- The model may require careful tuning of loss weights and training strategy.

## Status

Reproduced.
