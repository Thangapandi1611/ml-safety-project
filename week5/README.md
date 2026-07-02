# Week 5 – Robustness Testing and K-Projection Coverage Analysis

## Overview

This week evaluates the robustness of the previously trained ResNet18 binary classifiers under different driving conditions. The pedestrian, vehicle, and traffic-light models are tested on multiple unseen datasets containing fog, night, and different town environments.

Additionally, K-Projection Coverage analysis is performed to measure how well the designed test scenarios cover the Operational Design Domain (ODD).

---

## Folder Contents

- `Testing_Pedestrain_on_different_type_of_data.ipynb`
- `Testing_Vehicle_model_on_different_types_of_data.ipynb`
- `Testing_Traffic_light_on_different_type_of_data.ipynb`
- `KProjection.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision pandas numpy matplotlib pillow scikit-learn seaborn
```

---

## Dataset

The notebooks expect the following datasets:

```
test-fog/
test-night/
test-town-01/
```

Each dataset should contain:

```
labels.csv
rgb-front/
```

The trained model files should also be available:

```
pedestrian_model.pth
Vehicle_model.pth
Traffic_Light_model.pth
```

Update all dataset and model paths before execution.

---

## How to Reproduce the Results

### 1. Pedestrian Model Robustness Test

Run:

```
Testing_Pedestrain_on_different_type_of_data.ipynb
```

This notebook:

- Loads the trained pedestrian model
- Evaluates the model on:
  - Fog dataset
  - Night dataset
  - Town-01 dataset
- Computes:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

### 2. Vehicle Model Robustness Test

Run:

```
Testing_Vehicle_model_on_different_types_of_data.ipynb
```

The notebook evaluates the vehicle classifier on the same three datasets and reports the same evaluation metrics.

---

### 3. Traffic Light Model Robustness Test

Run:

```
Testing_Traffic_light_on_different_type_of_data.ipynb
```

This notebook evaluates the traffic-light classifier on the three unseen datasets and compares its performance under different environmental conditions.

---

### 4. K-Projection Coverage Analysis

Run:

```
KProjection.ipynb
```

This notebook:

- Defines the Operational Design Domain (ODD)
- Creates driving scenarios
- Computes:
  - 1-Projection Coverage
  - 2-Projection Coverage
  - 3-Projection Coverage

The coverage results indicate how well the selected test scenarios represent the possible combinations within the defined ODD.

---

## Expected Outputs

Running all notebooks will produce:

- Performance evaluation for pedestrian model
- Performance evaluation for vehicle model
- Performance evaluation for traffic-light model
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Comparison tables
- K-Projection coverage values for k = 1, 2 and 3

---

## Notes

- All evaluations use the previously trained ResNet18 models from Week 4.
- Images are resized to **224 × 224** before inference.
- The same evaluation pipeline is used for all three classifiers.
- K-Projection Coverage is included to assess the completeness of the selected test scenarios for autonomous driving applications.
