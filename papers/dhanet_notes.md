# DHANet Notes

## Paper

**Title:** A Novel Dynamic Neural Network for Heterogeneity-Aware Structural Brain Network Exploration and Alzheimer's Disease Diagnosis

## Modality

- Structural MRI

## Main Task

The model is designed for Alzheimer's disease diagnosis using structural MRI data.

Main classification tasks in my reproduction include:

- AD vs CN
- AD vs MCI
- MCI vs CN

## Core Idea

DHANet focuses on heterogeneity-aware Alzheimer's disease diagnosis.

In Alzheimer's disease, different patients may show different brain atrophy patterns. A fixed CNN may not fully capture these patient-specific differences.

DHANet tries to model this heterogeneity by using dynamic convolution, prototype learning, and graph-based brain network modeling.

In simple terms, DHANet does not treat every subject with exactly the same feature extraction strategy. Instead, it tries to adapt to different structural brain patterns.

## Main Components

### 1. 3D Dynamic Convmixer

DHANet uses a 3D dynamic convolution-based feature extractor.

Compared with a standard CNN, dynamic convolution can adapt its convolution behavior according to the input subject.

This is useful because different patients may have different disease-related structural changes.

### 2. Dynamic Region-Aware Convolution

The model uses region-aware dynamic convolution to capture local disease-related brain patterns.

This helps the model focus on subject-specific and region-specific structural abnormalities.

### 3. Hierarchical Prototype Learning

DHANet learns hierarchical prototypes to represent different disease-related patterns.

These prototypes can be understood as representative structural patterns shared by groups of subjects.

### 4. Contrastive Learning

The model uses contrastive learning to encourage meaningful feature organization.

Similar disease-related patterns are pulled closer in feature space, while different patterns are pushed apart.

### 5. Joint Dynamic Edge Correlation

DHANet builds a dynamic brain network by estimating relationships between brain regions.

Instead of using a fixed brain graph for every subject, the model learns subject-specific structural relationships.

### 6. Graph Convolutional Network

The learned brain network is processed by a graph convolutional network.

The GCN helps model relationships among brain regions and combines regional information for final classification.

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
| AD vs CN | 88.4 | 77.6 | 93.2 | 94.8 | 80.3 | 84.7 |
| AD vs MCI | 75.5 | 53.8 | 86.9 | 81.3 | 59.7 | 69.1 |
| MCI vs CN | 55.7 | 8.4 | 95.9 | 64.0 | 10.1 | 12.7 |

## Observations

1. DHANet performs well on AD vs CN, showing that dynamic structural brain modeling is useful.
2. AD vs MCI is more difficult because MCI is a transitional disease stage.
3. MCI vs CN shows very low sensitivity in my reproduction, meaning the model often predicts MCI subjects as CN.
4. The dynamic graph design is interesting, but it also makes the model harder to train and tune.
5. Heterogeneity-aware modeling is clinically meaningful because AD patients may not share exactly the same atrophy pattern.

## Reproduction Challenges

- Dynamic convolution is more complex than standard convolution.
- Prototype learning requires careful implementation and hyperparameter tuning.
- Graph construction can strongly affect final performance.
- MCI vs CN is difficult due to subtle disease patterns and class overlap.
- Training the full model can be computationally expensive.

## Status

Reproduced.
