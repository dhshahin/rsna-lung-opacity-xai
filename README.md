# RSNA Lung Opacity Reliability & Explainable AI

`v0.1 — Pre-ISBI-extension reproducible snapshot`

This repository contains a cleaned snapshot of a multiclass chest X-ray study built on the RSNA Pneumonia Detection Challenge data. The project evolved from a 2024 MobileNetV2 project into a 2026 reliability and explainability study focused on acquisition view position, subgroup robustness, and quantitative XAI.

## Research question
How robust is multiclass chest X-ray classification to AP/PA acquisition position, and how stable are lesion-localization conclusions across attribution methods?

The task retains the original classes: `Normal`, `No Lung Opacity / Not Normal`, and `Lung Opacity`. The repository uses **Lung Opacity** rather than claiming direct clinical pneumonia diagnosis.

## Current locked study state
The fixed 70/15/15 split contains 26,684 unique frontal radiographs. Model and XAI selection were performed on validation data before the internal test evaluation.

### Primary locked model
Multi-scale MobileNetV2 features + multi-head self-attention + residual refinement.

| Metric | Locked test |
|---|---:|
| Accuracy | 0.7050 |
| Balanced accuracy | 0.6844 |
| Macro F1 | 0.6892 |
| Macro ROC-AUC | 0.8582 |
| Macro PR-AUC | 0.7420 |

### Main reliability finding
On the locked test set, Lung Opacity recall was 0.617 for AP and 0.144 for PA images. This snapshot predates the planned operating-point extension.

### XAI state
LayerCAM at `multiscale_fusion` was selected on validation data. Integrated Gradients with 256 steps and a zero preprocessed baseline was retained as a complementary method.

| Method | Mean enrichment | Pointing game |
|---|---:|---:|
| Position-conditioned spatial prior | 2.776 | 0.599 |
| LayerCAM | 5.764 | 0.713 |
| Integrated Gradients 256 | 1.809 | 0.484 |

This supports a narrow claim of radiologist-box localization beyond a fixed spatial prior under the specified implementation; it is not evidence of causal model reasoning.

## Reproducibility
1. `pip install -r requirements.txt`
2. Obtain the data with `kagglehub.dataset_download("iamtapendu/rsna-pneumonia-processed-dataset")`.
3. Optionally set `RSNA_PROJECT_ROOT` to a persistent experiment directory.
4. Run `notebooks/rsna_multiclass_xai_reproducible_snapshot.ipynb` in order.

Raw images, model weights, checkpoints, caches, and large per-case prediction tables are intentionally not versioned.

## Planned ISBI extension
Planned, not completed in v0.1:
- validation-derived matched-specificity / view-conditioned operating-point analysis;
- recovered-vs-still-missed opacity analysis linked to locked XAI maps;
- AP/PA confusion flow and architecture robustness across existing models;
- per-view calibration and logit-probability sensitivity analysis;
- possible bidirectional acquisition-style perturbation probe.

## Intended use
Research and reproducibility only. Not a clinical diagnostic tool.

## License
Code: MIT. Dataset terms remain governed by the original RSNA/Kaggle sources.
