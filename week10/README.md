# Week 9 – Adversarial Robustness with FGSM

## Overview

This week explores adversarial robustness using the Fast Gradient Sign Method (FGSM). The experiments generate adversarial examples for the trained perception models and evaluate how increasing perturbation strength affects model performance.

The robustness of the pedestrian, vehicle and traffic-light classifiers is investigated by measuring changes in recall under different FGSM attack strengths.

---

## Folder Contents

- `FGSM.ipynb`
- `FGSM_RECALL_TRADEOFF.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib pillow opencv-python scikit-learn
```

---

## Dataset

The notebooks require:

```
test/

rgb-front/
labels.csv
```

Pre-trained models:

```
pedestrian_model.pth
Vehicle_model.pth
Traffic_Light_model.pth
```

Update all dataset and model paths before running the notebooks.

---

## How to Reproduce the Results

### 1. FGSM Adversarial Attack

Run:

```
FGSM.ipynb
```

This notebook:

- Loads the trained pedestrian, vehicle and traffic-light ResNet18 classifiers.
- Selects correctly classified test images.
- Computes image gradients using binary cross-entropy loss.
- Generates adversarial examples using the Fast Gradient Sign Method (FGSM).
- Applies multiple perturbation strengths (ε = 0.01, 0.05, 0.10).
- Visualizes the clean image alongside its adversarial versions to demonstrate how small perturbations can influence model predictions.

---

### 2. FGSM Recall Trade-off

Run:

```
FGSM_RECALL_TRADEOFF.ipynb
```

This notebook:

- Loads each trained classifier.
- Randomly samples images from the test dataset.
- Evaluates baseline (clean) recall.
- Generates FGSM adversarial examples for multiple epsilon values.
- Computes recall after each attack strength.
- Compares recall degradation across different perturbation levels to analyze model robustness.

---

## Expected Outputs

Running both notebooks will produce:

- Clean and adversarial image visualizations.
- FGSM attacks with multiple epsilon values.
- Recall scores for clean and adversarial datasets.
- Comparison of recall under increasing attack strengths.
- Robustness analysis for the pedestrian, vehicle and traffic-light classifiers.

---

## Notes

- All experiments use the previously trained ResNet18 binary classifiers.
- Images are resized to **224 × 224** before inference.
- FGSM generates adversarial examples using the sign of the input gradient.
- Perturbation strengths of **ε = 0.01, 0.05, and 0.10** are evaluated.
- Recall is used as the primary evaluation metric to measure the effect of adversarial attacks on safety-critical object detection.
