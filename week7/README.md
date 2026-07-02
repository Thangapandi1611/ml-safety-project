# Week 7 – Explainability and Occlusion Analysis

## Overview

This week focuses on understanding and interpreting the decisions made by the trained ResNet18 classifiers using Occlusion Sensitivity.

Occlusion heatmaps are generated to identify which image regions contribute most to the model's predictions. The effect of occluding important regions is also evaluated on the vehicle classifier using the night dataset.

---

## Folder Contents

- `Explainability_Occulation.ipynb`
- `Occulation_Vehicle_night_.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision pandas numpy matplotlib pillow opencv-python scikit-learn seaborn
```

---

## Dataset

The notebooks require:

```
test/
rgb-front/
labels.csv

test-night/
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

### 1. Explainability using Occlusion Sensitivity

Run:

```
Explainability_Occulation.ipynb
```

This notebook:

- Loads the pedestrian, vehicle and traffic-light classifiers.
- Selects sample images from the CARLA dataset.
- Applies Occlusion Sensitivity by sliding a masking window across the image.
- Measures the change in prediction confidence after masking each region.
- Generates heatmaps highlighting the regions most important for each prediction.

---

### 2. Occlusion Analysis on Night Dataset

Run:

```
Occulation_Vehicle_night_.ipynb
```

This notebook:

- Loads the trained vehicle detection model.
- Evaluates model performance on the night dataset.
- Generates occlusion heatmaps.
- Applies occlusions to important image regions.
- Measures how occlusions affect model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## Expected Outputs

Running all notebooks will produce:

- Occlusion sensitivity heatmaps.
- Visual explanation of model predictions.
- Performance metrics before and after occlusion.
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices.

---

## Notes

- All experiments use the previously trained ResNet18 models.
- Images are resized to **224 × 224** before inference.
- Occlusion Sensitivity is used as an Explainable AI (XAI) technique to identify the image regions that most influence the model's predictions.
- The night dataset is used to study how occluding important regions impacts model performance under challenging driving conditions.
