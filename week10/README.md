````markdown
# Week 10 – Model Calibration and Cost-Optimal Decision Making

## Overview

This week investigates the calibration of the trained CARLA perception models and examines how calibrated confidence scores can be used for cost-sensitive decision making.

The experiments cover:

1. Measuring calibration using Expected Calibration Error (ECE).
2. Generating reliability diagrams.
3. Applying temperature scaling to calibrate model probabilities.
4. Finding the temperature that minimizes Negative Log-Likelihood (NLL).
5. Comparing calibration before and after temperature scaling.
6. Evaluating a cost-sensitive decision threshold for the pedestrian detector.

The experiments are performed on the pedestrian, vehicle and traffic-light perception models trained in the previous weeks.

---

## Folder Contents

- `calibration.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn seaborn scipy
````

---

## Dataset

The notebook requires the CARLA test dataset:

```text
test/
│
├── labels.csv
└── rgb-front/
```

Download the dataset from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

Update the dataset path in the notebook before running the experiment.

The notebook uses:

```text
/content/drive/MyDrive/MLS/test
```

---

## Pre-trained Models

The notebook uses the trained models from the previous weeks:

```text
pedestrian_model.pth
Vehicle_model.pth
Traffic_Light_model.pth
```

Download the trained models from:

[https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing](https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing)

Update the model paths before running the notebook.

---

## Model Setup

All three models use the previously trained ResNet18 architecture.

The final fully connected layer is replaced with a single output:

```python
model.fc = nn.Linear(
    model.fc.in_features,
    1
)
```

The trained state dictionary is then loaded into the model.

Images are resized to:

```text
224 × 224
```

using:

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
```

The notebook automatically uses CUDA when available.

---

## How to Reproduce the Results

### 1. Baseline Model Evaluation

The notebook first evaluates the trained models using the standard binary classification setup.

For each model:

* The test dataset is loaded.
* The trained ResNet18 model is loaded.
* Sigmoid converts the model output into a probability.
* A threshold of 0.5 is used for the binary prediction.
* Accuracy, precision, recall and F1-score are calculated.
* A confusion matrix is generated.

The evaluation function is:

```text
evaluate_model()
```

The three target labels are:

```text
Pedestrian    → has_pedestrian
Vehicle       → has_vehicle
Traffic Light → has_traffic_light
```

---

### 2. Expected Calibration Error (ECE)

The notebook implements Expected Calibration Error using:

```text
compute_ece()
```

The predicted probabilities are divided into 10 confidence bins.

For each bin, the notebook calculates:

* Average prediction confidence.
* Average observed accuracy.

The calibration error is then calculated as the weighted difference between confidence and accuracy.

The implementation uses:

```python
compute_ece(
    y_true,
    y_prob,
    n_bins=10
)
```

A lower ECE indicates better agreement between the model's predicted confidence and its observed accuracy.

---

### 3. Reliability Diagram

The notebook generates a reliability diagram using:

```text
plot_reliability_diagram()
```

The plot contains:

* A diagonal reference line representing perfect calibration.
* The model's confidence-versus-accuracy curve.

The axes are:

```text
X-axis → Confidence
Y-axis → Accuracy
```

A model that is well calibrated should have its reliability curve close to the perfect-calibration diagonal.

---

### 4. Calibration of the Three Models

The notebook evaluates calibration separately for:

```text
Pedestrian model
Vehicle model
Traffic-light model
```

The calibration function:

```text
evaluate_calibration()
```

collects the model's sigmoid probabilities and ground-truth labels, calculates ECE and generates the corresponding reliability diagram.

The output includes:

```text
ECE
Reliability Diagram
```

for each model.

---

## Temperature Scaling

### 5. Finding the Optimal Temperature

The notebook then applies temperature scaling to the model logits.

The temperature-scaled probability is calculated as:

```text
probability = sigmoid(logit / T)
```

where `T` is the temperature parameter.

The notebook searches for the temperature that minimizes:

```text
Negative Log-Likelihood (NLL)
```

The optimisation is performed using:

```text
scipy.optimize.minimize
```

with the temperature constrained to:

```text
0.05 ≤ T ≤ 10
```

The initial temperature is:

```text
T = 1.0
```

---

### 6. Temperature Optimization

The NLL objective is implemented using:

```text
temperature_nll()
```

For a given temperature:

1. The logits are divided by `T`.
2. Sigmoid converts them into probabilities.
3. Probabilities are clipped to avoid numerical issues.
4. Binary cross-entropy/NLL is calculated.
5. The optimizer searches for the temperature that minimizes this value.

The optimal temperature is obtained using:

```text
find_temperature()
```

---

### 7. Calibration Before and After Temperature Scaling

For each model, the notebook calculates:

```text
ECE Before
ECE After
Best Temperature
```

The procedure is:

```text
Original logits
      ↓
Original probabilities
      ↓
ECE Before
      ↓
Find optimal T using NLL
      ↓
Scale logits by T
      ↓
Calibrated probabilities
      ↓
ECE After
```

A reduction in ECE after temperature scaling indicates improved probability calibration.

---

## Models Used for Temperature Scaling

### Pedestrian Model

```text
Model:
pedestrian_model.pth

Label:
has_pedestrian
```

The notebook calculates:

```text
Best T
ECE Before
ECE After
```

for the pedestrian detector.

---

### Vehicle Model

```text
Model:
Vehicle_model.pth

Label:
has_vehicle
```

The same temperature-scaling procedure is applied to the vehicle detector.

---

### Traffic-Light Model

```text
Model:
Traffic_Light_model.pth

Label:
has_traffic_light
```

The same procedure is applied to the traffic-light detector.

---

# Cost-Optimal Decision in Practice

## 8. Cost-Sensitive Decision Making

The notebook also implements a cost-sensitive decision function:

```text
compute_cost_loss()
```

This experiment focuses on the pedestrian detector.

The purpose is to demonstrate that the best classification threshold depends on the relative cost of false negatives and false positives.

The selected costs are:

```text
False Negative = 100
False Positive = 1
```

Therefore:

```text
Cost = 100 × FN + 1 × FP
```

A false negative is intentionally assigned a much higher cost because missing a pedestrian is more safety-critical than generating a false pedestrian detection.

---

## 9. Decision Thresholds

The notebook compares four configurations.

### Case 1 — Uncalibrated, standard threshold

```text
Temperature = None
Threshold = 0.5
```

### Case 2 — Uncalibrated, cost-oriented threshold

```text
Temperature = None
Threshold = 0.0099
```

### Case 3 — Calibrated, standard threshold

```text
Temperature = 1.9505
Threshold = 0.5
```

### Case 4 — Calibrated, cost-oriented threshold

```text
Temperature = 1.9505
Threshold = 0.0099
```

For each configuration, the notebook reports:

```text
Total Cost
False Positives
False Negatives
```

---

## 10. Cost Calculation

The cost is calculated from the confusion matrix:

```python
loss = (
    100 * fn
    +
    1 * fp
)
```

where:

```text
FN = False Negatives
FP = False Positives
```

The objective is therefore to reduce the number of expensive false-negative pedestrian detections while accounting for the additional false positives produced by a lower decision threshold.

---

## Expected Outputs

Running `calibration.ipynb` will produce:

* Baseline classification metrics.
* Confusion matrices.
* ECE values.
* Reliability diagrams.
* Optimal temperature values.
* ECE before temperature scaling.
* ECE after temperature scaling.
* Cost values for different decision thresholds.
* False-positive counts.
* False-negative counts.

---

## Notes

* All experiments use the previously trained ResNet18 perception models.
* Images are resized to **224 × 224** before inference.
* Binary probabilities are obtained using the sigmoid function.
* The initial classification threshold is **0.5**.
* ECE uses **10 confidence bins**.
* Temperature scaling optimizes NLL.
* The temperature search is bounded between **0.05 and 10**.
* The cost-sensitive experiment focuses on the **pedestrian model**.
* The selected safety-oriented costs are **100 for a false negative** and **1 for a false positive**.
* The cost-sensitive experiment compares both calibrated and uncalibrated predictions.
* The notebook uses a calibrated temperature of **1.9505** and a cost-oriented threshold of **0.0099** in the final comparison.

---


```
```
