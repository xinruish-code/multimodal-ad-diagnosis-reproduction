# Multimodal Alzheimer's Disease Diagnosis Reproduction

This repository contains my reproduction and adaptation of recent deep learning methods for Alzheimer's disease diagnosis using structural MRI, PET, and multimodal neuroimaging data.

The project focuses on binary and multiclass classification tasks among Alzheimer's disease (AD), mild cognitive impairment (MCI), and cognitively normal (CN) subjects.

## Project Overview

This repository currently covers 7 reproduced or ongoing reproduction papers:

### Previous Reproduced Papers

1. **MDL-Net**  
   *3D Multimodal Fusion Network With Disease-Induced Joint Learning for Early Alzheimer's Disease Diagnosis*  
   Modality: sMRI + PET  
   Main idea: multimodal 3D feature fusion with disease-induced region-aware learning.

2. **MMRN**  
   *Multi-Template Meta-Information Regularized Network for Alzheimer's Disease Diagnosis Using Structural MRI*  
   Modality: sMRI  
   Main idea: multi-template self-supervised learning with meta-information disentanglement.

3. **LA-GMF**  
   *Interpretable Medical Deep Framework by Logits-Constraint Attention Guiding Graph-Based Multi-Scale Fusion for Alzheimer's Disease Analysis*  
   Modality: sMRI  
   Main idea: graph-based multi-scale fusion with logits-constrained attention.

4. **DHANet**  
   *A Novel Dynamic Neural Network for Heterogeneity-Aware Structural Brain Network Exploration and Alzheimer's Disease Diagnosis*  
   Modality: sMRI  
   Main idea: dynamic convolution, hierarchical prototype learning, and graph-based structural brain network modeling.

5. **AAGN**  
   *Anatomy-Aware Gating Network for Explainable Alzheimer's Disease Diagnosis*  
   Modality: sMRI  
   Main idea: anatomy-aware ROI feature extraction and differentiable ROI selection for explainable diagnosis.

### Current Multimodal Reproduction Papers

6. **CGAN-AD**  
   *A coupled-GAN architecture to fuse MRI and PET image features for multi-stage classification of Alzheimer's disease*  
   Modality: MRI + PET  
   Main idea: coupled-GAN-based feature fusion using shared latent representations, reconstruction, and adversarial learning.

7. **UniCross**  
   *Balanced Multimodal Learning for Alzheimer's Disease Diagnosis by Uni-modal Separation and Metadata-guided Cross-modal Interaction*  
   Modality: sMRI + FDG-PET + metadata  
   Main idea: balanced multimodal learning to reduce modality laziness through uni-modal separation, shared heads, and metadata-weighted contrastive learning.

---

## Classification Tasks

The main classification tasks include:

- AD vs CN
- AD vs MCI
- MCI vs CN
- AD vs MCI vs CN

Some reproduced papers were originally designed for binary classification, while others support multiclass or multimodal diagnosis. When necessary, the original code was adapted to pairwise AD/MCI/CN classification tasks.

---

## Modalities

This project covers the following neuroimaging and clinical modalities:

- T1-weighted structural MRI
- Gray matter and white matter tissue maps
- PET
- FDG-PET
- Tau PET
- Optional metadata such as age, sex, education, and cognitive scores
- Optional clinical tabular features such as MoCA

---

## Reproduced Models

| Model | Paper Type | Modality | Main Method | Task Type |
|---|---|---|---|---|
| MDL-Net | Journal | sMRI + PET | Multi-fusion joint learning + disease-induced ROI learning | Binary classification |
| MMRN | Journal | sMRI | Multi-template SSL + meta-information disentanglement | Binary classification |
| LA-GMF | Journal / Conference | sMRI | Graph-based multi-scale fusion + logits-constrained attention | Binary classification |
| DHANet | Journal | sMRI | Dynamic convolution + hierarchical graph representation | Binary classification |
| AAGN | MICCAI | sMRI | Anatomy-aware squeeze-and-excite + ROI gating | Binary classification |
| CGAN-AD | Journal | MRI + PET | Coupled-GAN feature fusion | Multistage AD classification |
| UniCross | MICCAI | sMRI + PET + metadata | Uni-modal separation + metadata-guided contrastive learning | AD diagnosis / MCI conversion |

---

## Summary of Previous Reproduction Results

### AD vs CN

| Model | ACC (%) | SEN (%) | SPE (%) | AUC (%) | Modality |
|---|---:|---:|---:|---:|---|
| MDL-Net | 88.5 | 77.6 | 93.3 | 96.1 | sMRI + PET |
| MMRN | 88.5 | 73.6 | 95.1 | 94.4 | sMRI |
| LA-GMF | 93.1 | 95.8 | 87.0 | 94.6 | sMRI |
| DHANet | 88.4 | 77.6 | 93.2 | 94.8 | sMRI |
| AAGN | 87.1 | 74.5 | 92.7 | 93.4 | sMRI |

### AD vs MCI

| Model | ACC (%) | SEN (%) | SPE (%) | AUC (%) | Modality |
|---|---:|---:|---:|---:|---|
| MDL-Net | 78.3 | 60.6 | 87.5 | 85.3 | sMRI + PET |
| MMRN | 78.6 | 52.8 | 92.0 | 81.9 | sMRI |
| LA-GMF | 80.0 | 85.3 | 69.8 | 82.7 | sMRI |
| DHANet | 75.5 | 53.8 | 86.9 | 81.3 | sMRI |
| AAGN | 76.0 | 49.8 | 89.7 | 81.5 | sMRI |

### CN vs MCI

| Model | ACC (%) | SEN (%) | SPE (%) | AUC (%) | Modality | Note |
|---|---:|---:|---:|---:|---|---|
| MDL-Net | 66.4 | 44.4 | 83.9 | 75.0 | sMRI + PET | 9/10 folds due to HPC limit |
| MMRN | 59.0 | 53.6 | 63.6 | 63.6 | sMRI | Single split |
| LA-GMF | 67.1 | 56.3 | 76.3 | 68.0 | sMRI | 3-fold CV |
| DHANet | 55.7 | 8.4 | 95.9 | 64.0 | sMRI | Low MCI sensitivity |
| AAGN | 63.5 | 58.8 | 67.5 | 68.2 | sMRI | 5-fold CV |

---

## Key Observations

1. **AD vs CN is the easiest task.**  
   Most models achieve strong AUC scores above 93%, suggesting that structural and multimodal imaging features can capture clear disease-related differences between AD and CN.

2. **MCI-related tasks are much harder.**  
   AD vs MCI and MCI vs CN show lower sensitivity and higher variance because MCI is a heterogeneous prodromal stage with subtle and overlapping patterns.

3. **PET improves multimodal diagnosis.**  
   MDL-Net achieves the highest AD vs CN AUC among the previous reproductions, likely because PET provides complementary metabolic information beyond sMRI.

4. **Interpretability is important.**  
   AAGN and MDL-Net provide anatomy-aware or ROI-aware explanations, making them more suitable for clinical interpretation than purely black-box models.

5. **Balanced multimodal learning is a key direction.**  
   CGAN-AD and UniCross focus more directly on multimodal feature fusion, cross-modal alignment, and the problem of strong-modality dominance.

---

## Repository Structure

```text
multimodal-ad-diagnosis/
├── README.md
├── papers/
│   ├── mdl_net_notes.md
│   ├── mmrn_notes.md
│   ├── la_gmf_notes.md
│   ├── dhanet_notes.md
│   ├── aagn_notes.md
│   ├── cgan_ad_notes.md
│   └── unicross_notes.md
│
├── preprocessing/
│   ├── preprocess_mri.py
│   ├── preprocess_pet.py
│   └── make_csv.py
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── configs/
│   ├── mdl_net.yaml
│   ├── mmrn.yaml
│   ├── la_gmf.yaml
│   ├── dhanet.yaml
│   ├── aagn.yaml
│   ├── cgan_ad.yaml
│   └── unicross.yaml
│
├── results/
│   ├── tables/
│   └── figures/
│
└── notebooks/
    ├── data_check.ipynb
    └── result_analysis.ipynb
```
---

## Data Notice

The original medical imaging data are not included in this repository due to data usage agreements, privacy restrictions, and file size limitations.

Users should download the required data from authorized sources such as ADNI and organize the files according to the provided CSV templates.

Do not upload raw medical imaging data or patient-level data to this repository, including but not limited to:

- `.nii`
- `.nii.gz`
- `.npy`
- `.npz`
- `.h5`
- `.hdf5`
- raw MRI/PET images
- patient identifiers
- protected health information
- model checkpoints trained on restricted data

---

## Reproduction Notes

This project is a research-oriented reproduction repository. Some experiments are direct reproductions of the original papers, while others are adapted to available ADNI data and pairwise AD/MCI/CN classification settings.

Differences from the original papers may include:

- different train/validation/test splits
- different cross-validation settings
- limited GPU time
- missing modalities for some subjects
- modified label mappings for pairwise classification
- adapted preprocessing pipelines
- different hardware environments
- simplified or partial reproduction of some modules

Therefore, the reproduced numbers should be interpreted as implementation-level reproduction results rather than exact official benchmarks.

---

## Future Work

Planned improvements include:

- organizing code for all reproduced models into a unified structure
- adding configuration files for each model
- improving data preprocessing scripts
- adding standardized evaluation metrics
- adding experiment logs and result tables
- comparing multimodal fusion strategies across models
- improving MCI classification using class balancing or focal loss
- analyzing model interpretability with ROI-level or attention-based explanations

---

## Author

Xinrui Shen  
Applied Mathematical Data Science  
University of Southern California
