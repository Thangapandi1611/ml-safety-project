# Week 6 – Model Security and Confidence Calibration

## Overview

This week investigates the reliability and security of the trained pedestrian detection model through two complementary experiments.

The first notebook demonstrates a data poisoning (backdoor) attack by inserting trigger patterns into images and evaluating the model's vulnerability using Attack Success Rate (ASR).

The second notebook studies confidence calibration using Temperature Scaling, analyzing how different temperature values affect the model's output probabilities while preserving prediction performance.

---

## Folder Contents

- `PoisoningData.ipynb`
- `Ped_model_with_Temp.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision pandas numpy matplotlib pillow scikit-learn seaborn
```

---

## Dataset

The notebooks expect:

```
test/
labels.csv
rgb-front/
```

Pre-trained model:

```
pedestrian_model.pth
```

Update all dataset and model paths before running the notebooks.

---

## How to Reproduce the Results

### 1. Data Poisoning (Backdoor Attack)

Run:

```
PoisoningData.ipynb
```

This notebook:

- Loads the trained pedestrian detection model.
- Adds a trigger pattern (red square) to selected pedestrian images.
- Modifies a portion of the dataset to simulate a poisoning attack.
- Evaluates the model using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Computes the Attack Success Rate (ASR) to measure the effectiveness of the backdoor attack.

---

### 2. Temperature Scaling

Run:

```
Ped_model_with_Temp.ipynb
```

This notebook evaluates the model using multiple temperature values:

- T = 0.5
- T = 1.0
- T = 2.0

For each temperature, the notebook:

- Applies temperature scaling to the model logits.
- Computes:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Visualizes the probability distribution of model predictions.

---

## Expected Outputs

Running all notebooks will produce:

- Performance metrics under poisoning attacks.
- Attack Success Rate (ASR).
- Confusion matrices.
- Performance comparison across different temperature values.
- Probability distribution histograms.
- Analysis of model confidence calibration.

---

## Notes

- All experiments use the previously trained ResNet18 pedestrian
