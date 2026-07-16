# E2AD Notes

## Paper

**Title:** Enhanced and Explainable Alzheimer's Disease Detection Framework via Anatomy- and Relation-aware Cross-modal Knowledge Distillation

**Short name:** E2AD / E²AD

**Venue:** Medical Image Analysis, 2026

**Code:** https://github.com/thibault-wch/E2AD-for-Alzheimer-disease

## Modality

- Training stage: paired MRI + PET
- Inference stage: MRI only

## Main Task

E2AD is designed for Alzheimer's disease detection and MCI progression prediction under a missing-modality setting.

Main tasks include:

- NC vs AD diagnosis
- sMCI vs pMCI progression prediction

The core clinical motivation is that PET provides useful metabolic information, but PET is expensive and often unavailable. Therefore, the model learns from MRI+PET during training and performs MRI-only inference at deployment.

## Core Idea

E2AD uses a multimodal MRI+PET teacher to guide an MRI-only student.

Instead of distilling only final logits, E2AD transfers three kinds of knowledge:

1. **Logit knowledge:** the student learns the teacher's soft prediction distribution.
2. **Anatomical knowledge:** the student learns which anatomical regions the teacher considers important.
3. **Relation knowledge:** the student learns the teacher's subject-subject relationship structure in feature space.

In simple terms, E2AD tries to answer this question:

> Can an MRI-only model learn the diagnostic behavior, anatomical attention, and cohort-level relation structure of an MRI+PET model?

## Pipeline

```text
Stage 1: Multimodal Teacher
MRI + PET
→ encoder
→ anatomical tokenizer
→ anatomy-aware feature modeling
→ differential anatomical router
→ classifier

Stage 2: MRI-only Student
MRI only
→ same anatomy-aware architecture
→ distillation from teacher
→ MRI-only diagnosis
```

## Main Components

### 1. Multimodal Teacher

The teacher uses both MRI and PET as input. MRI provides structural information such as brain atrophy, while PET provides functional or metabolic information.

The teacher acts as the upper-bound multimodal model because it has access to richer information than the MRI-only student.

### 2. Anatomical Tokenizer

The anatomical tokenizer converts image-level feature maps into ROI-level anatomical tokens using atlas-based brain regions.

This changes the representation from a raw 3D feature map into anatomical-region-level features.

This is useful because Alzheimer's disease is strongly related to specific brain regions, such as the hippocampus, parahippocampal gyrus, and temporal regions.

### 3. Anatomy-aware Feature Modeling

E2AD separates and models anatomical information at the ROI level.

The model tries to preserve both:

- shared disease patterns across anatomical regions
- ROI-specific disease patterns

This makes the learned representation more interpretable than a global black-box feature vector.

### 4. Differential Anatomical Router

The differential anatomical router produces anatomical routing weights over brain regions.

These weights indicate which anatomical regions are more important for the current subject's prediction.

This module supports explainability because the model can report which ROIs contribute more strongly to the decision.

### 5. Cross-modal Knowledge Distillation

The MRI-only student is trained with three distillation losses:

- **LogitKD:** aligns student predictions with teacher predictions.
- **AnaKD:** aligns student anatomical routing weights with teacher anatomical routing weights.
- **RelKD:** aligns student subject-subject relation structure with the teacher's relation structure.

This is the main innovation of E2AD.

Rather than only asking the student to copy the teacher's final answer, E2AD also asks the student to copy:

1. where the teacher looks anatomically;
2. how the teacher organizes relationships among subjects.

## Why It Is a Missing-Modality Method

E2AD is a missing-modality method because the teacher uses MRI+PET during training, but the final deployed model only requires MRI.

```text
Training: MRI + PET available
Testing / deployment: MRI only
```

Therefore, PET knowledge is transferred to the MRI-only model through cross-modal distillation.

## Reproduction Setting

| Item | Setting |
|---|---|
| Teacher input | MRI + PET |
| Student input | MRI only |
| Missing modality | PET missing at inference |
| Main tasks | NC vs AD, sMCI vs pMCI |
| Core losses | CE + LogitKD + AnaKD + RelKD |
| Metrics to report | ACC, SEN, SPE, F1, AUC |

## Result Reporting Rule

Because E2AD is a teacher-student missing-modality method, both teacher and student results should be preserved.

The result table should include:

- MRI+PET teacher
- MRI-only student baseline without full distillation
- E2AD MRI-only student

Use mean ± standard deviation across folds or repeated runs.

| Task | Model Role | Method | Training Input | Inference Input | ACC | SEN | SPE | F1 | AUC |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| NC vs AD | Teacher | Multimodal teacher | MRI+PET | MRI+PET | TODO | TODO | TODO | TODO | TODO |
| NC vs AD | Student baseline | MRI-only baseline | MRI | MRI | TODO | TODO | TODO | TODO | TODO |
| NC vs AD | Student | E2AD | MRI+PET supervision | MRI | TODO | TODO | TODO | TODO | TODO |
| sMCI vs pMCI | Teacher | Multimodal teacher | MRI+PET | MRI+PET | TODO | TODO | TODO | TODO | TODO |
| sMCI vs pMCI | Student baseline | MRI-only baseline | MRI | MRI | TODO | TODO | TODO | TODO | TODO |
| sMCI vs pMCI | Student | E2AD | MRI+PET supervision | MRI | TODO | TODO | TODO | TODO | TODO |

## Key Strengths

1. Uses PET knowledge during training while requiring only MRI during inference.
2. Distills more than final logits by also transferring anatomical and relation-aware knowledge.
3. Provides ROI-level interpretability through anatomical routing weights.
4. Targets both AD diagnosis and MCI progression prediction.

## Limitations and Reproduction Challenges

- Requires paired MRI+PET data to train the teacher.
- Atlas registration and ROI extraction must be handled carefully.
- Anatomical routing weights are model-derived explanations, not causal medical proof.
- Relation-aware distillation may be sensitive to batch size and subject composition.
- Teacher and student results should be saved separately for fair comparison.

## Status

Paper notes prepared. Reproduction not yet completed.
