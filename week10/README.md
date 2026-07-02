# Week 10 – Model Calibration

## Overview

This week evaluates the calibration of the trained perception models. In addition to standard classification metrics, the experiments measure how well the predicted probabilities reflect the true likelihood of correct predictions using Expected Calibration Error (ECE) and Reliability Diagrams.

The pedestrian, vehicle and traffic-light classifiers are assessed to determine whether their confidence estimates are reliable for safety-critical autonomous driving applications.

---

## Folder Contents

- `calibration.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn pillow scikit-learn
```

---

## Dataset

The notebook requires:

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

Update all dataset and model paths before running the notebook.

---

## How to Reproduce the Results

Run:

```
calibration.ipynb
```

The notebook performs the following steps:

- Loads the trained ResNet18 classifier.
- Creates the CARLA test dataset.
- Performs inference on the test images.
- Computes classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Calculates the Expected Calibration Error (ECE).
- Generates a Reliability Diagram to compare prediction confidence with actual accuracy.
- Repeats the evaluation for the pedestrian, vehicle and traffic-light classifiers.

---

## Expected Outputs

Running the notebook will produce:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Expected Calibration Error (ECE)
- Reliability Diagram for each classifier
- Calibration analysis of prediction confidence

---

## Notes

- All experiments use the previously trained ResNet18 binary classifiers.
- Images are resized to **224 × 224** before inference.
- Model probabilities are obtained using the sigmoid activation function.
- Expected Calibration Error (ECE) is computed using confidence bins.
- Reliability Diagrams compare model confidence against observed accuracy.
- Lower ECE values indicate better-calibrated confidence estimates, making the models more reliable for safety-critical perception tasks.
