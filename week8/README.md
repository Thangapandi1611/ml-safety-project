# Week 8 – Distribution Shift Analysis and KNN-Based OOD Detection

## Overview

This week investigates model robustness under changing environmental conditions and introduces Out-of-Distribution (OOD) detection.

The experiments visualize distribution shifts across multiple CARLA datasets and implement a K-Nearest Neighbors (KNN)-based OOD detector using feature embeddings extracted from the trained ResNet18 vehicle classifier.

---

## Folder Contents

- `Visualising_the_Distribution_Shift.ipynb`
- `KNN_OOD_DECTIYION.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib pillow opencv-python scikit-learn seaborn
```

---

## Dataset

The notebooks require:

```
validation/
test/
test-fog/
test-night/
test-town-01/

rgb-front/
labels.csv
```

Pre-trained model:

```
Vehicle_model.pth
```

Update all dataset and model paths before running the notebooks.

---

## How to Reproduce the Results

### 1. Distribution Shift Visualization

Run:

```
Visualising_the_Distribution_Shift.ipynb
```

This notebook:

- Loads the trained pedestrian, vehicle and traffic-light classifiers.
- Visualizes images from different environmental conditions.
- Compares datasets collected under:
  - Sunny
  - Fog
  - Night
  - Different town
- Demonstrates how environmental changes introduce distribution shifts that can affect model performance.

---

### 2. KNN-Based Out-of-Distribution Detection

Run:

```
KNN_OOD_DECTIYION.ipynb
```

This notebook:

- Loads the trained vehicle classifier.
- Extracts deep feature embeddings from the ResNet18 backbone.
- Builds a K-Nearest Neighbors (KNN) model using validation features.
- Computes nearest-neighbor distances for in-distribution and out-of-distribution samples.
- Uses these distances as OOD scores.
- Evaluates the detector using ROC-AUC and visualizes the separation between ID and OOD samples.

---

## Expected Outputs

Running all notebooks will produce:

- Distribution shift visualizations.
- Deep feature embeddings.
- KNN distance-based OOD scores.
- ROC-AUC evaluation for OOD detection.
- Visual comparisons between in-distribution and out-of-distribution datasets.

---

## Notes

- All experiments use the previously trained ResNet18 vehicle classifier.
- Images are resized to **224 × 224** before inference.
- Deep feature embeddings are extracted from the penultimate layer of ResNet18.
- KNN-based distance serves as an uncertainty measure for detecting out-of-distribution samples.
- The experiments evaluate how environmental changes impact feature representations and model reliability.
