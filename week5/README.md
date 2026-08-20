````markdown
# Week 5 — Robustness and OOD Testing

This folder contains the Week 5 experiments for evaluating the CARLA perception models under different environmental and scenario conditions.

The experiments extend the Week 4 baseline models by testing them on data that differs from the original in-distribution test data.

The main experiments are:

1. Testing the pedestrian model on different types of data
2. Testing the vehicle model on different types of data
3. Testing the traffic-light model on different types of data
4. Measuring K-projection coverage of the test scenarios

---

## Folder Contents

```text
week5/
│
├── Testing_Pedestrain_on_different_type_of_data.ipynb
├── Testing_Vehicle_model_on_different_types_of_data.ipynb
├── Testing_Traffic_light_on_different_type_of_data.ipynb
├── KProjection.ipynb
└── README.md
````

---

# 1. Required Resources

The Week 5 experiments use the models trained during Week 4.

## Dataset

Download the required CARLA datasets from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

The experiments use the following test datasets:

```text
test-fog
test-night
test-town-01
```

The notebooks expect them under:

```text
/content/drive/MyDrive/MLS_DATA/
```

Therefore, the following directories should be available:

```text
/content/drive/MyDrive/MLS_DATA/test-fog
/content/drive/MyDrive/MLS_DATA/test-night
/content/drive/MyDrive/MLS_DATA/test-town-01
```

Each test dataset should contain its corresponding `labels.csv` and `rgb-front` image directory.

---

## Trained Models

Download the trained Week 4 models from:

[https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing](https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing)

The required model files are:

```text
pedestrian_model.pth
Vehicle_model.pth
Traffic_Light_model.pth
```

Place them under:

```text
/content/drive/MyDrive/MLS_DATA/
```

---

# 2. Common Evaluation Setup

The three perception models are evaluated using the same general procedure.

Each notebook:

* Mounts Google Drive
* Loads the corresponding trained model
* Loads foggy, nighttime, and different-town test datasets
* Resizes images to 224 × 224
* Uses the same CARLA dataset structure from Week 4
* Performs binary classification
* Calculates classification metrics
* Generates a comparison table
* Generates a performance comparison plot
* Generates confusion matrices

The image transformation is:

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
```

The evaluation uses:

```text
Batch size = 32
```

and automatically selects CUDA when available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

# 3. Testing the Pedestrian Model

## Notebook

```text
Testing_Pedestrain_on_different_type_of_data.ipynb
```

## Objective

The purpose of this experiment is to evaluate the Week 4 pedestrian perception model on different environmental and scenario conditions.

The trained model used is:

```text
pedestrian_model.pth
```

The target label is:

```text
has_pedestrian
```

---

## Test Conditions

The model is evaluated on three datasets:

```text
test-fog
test-night
test-town-01
```

These represent changes in:

* Weather conditions
* Lighting conditions
* CARLA town/environment

The experiment therefore checks whether the pedestrian model maintains its performance when the input distribution changes from the original test conditions.

---

## Evaluation Metrics

The notebook calculates:

* Accuracy
* Precision
* Recall
* F1-score

It also generates confusion matrices for each test condition.

---

## Comparison

The results from the three test conditions are combined into a pandas DataFrame with the following structure:

```text
Test
Accuracy
Precision
Recall
F1-score
```

A bar chart is then generated to compare the model performance across:

```text
Foggy Data
Night Data
Town Data
```

---

# 4. Testing the Vehicle Model

## Notebook

```text
Testing_Vehicle_model_on_different_types_of_data.ipynb
```

## Objective

This experiment evaluates the Week 4 vehicle perception model under different environmental and scenario conditions.

The trained model used is:

```text
Vehicle_model.pth
```

The target label is:

```text
has_vehicle
```

---

## Test Conditions

The model is evaluated using:

```text
test-fog
test-night
test-town-01
```

The experiment therefore examines whether the vehicle model remains reliable when the input conditions differ from the original test distribution.

---

## Evaluation Metrics

The following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-score

Confusion matrices are also generated for the individual test conditions.

---

## Comparison

The three test conditions are combined into a comparison table containing:

```text
Test
Accuracy
Precision
Recall
F1-score
```

A bar chart is generated to visualise the performance differences between foggy, nighttime, and different-town data.

---

# 5. Testing the Traffic-Light Model

## Notebook

```text
Testing_Traffic_light_on_different_type_of_data.ipynb
```

## Objective

This experiment evaluates the Week 4 traffic-light perception model on different environmental and scenario conditions.

The trained model used is:

```text
Traffic_Light_model.pth
```

The target label is:

```text
has_traffic_light
```

---

## Test Conditions

The model is evaluated using:

```text
test-fog
test-night
test-town-01
```

The purpose is to determine how the traffic-light detector behaves when weather, lighting, or the CARLA environment changes.

---

## Evaluation Metrics

The notebook calculates:

* Accuracy
* Precision
* Recall
* F1-score

Confusion matrices are also generated.

---

## Comparison

The three test conditions are compared using a pandas DataFrame containing:

```text
Test
Accuracy
Precision
Recall
F1-score
```

A bar chart is generated to compare the model's performance across the three conditions.

---

# 6. K-Projection Coverage

## Notebook

```text
KProjection.ipynb
```

## Objective

The K-projection experiment measures how much of the defined Operational Design Domain (ODD) combination space is actually represented by the test scenarios.

The ODD is defined using three dimensions:

```text
Weather
Lighting
Road Type
```

The values used in the notebook are:

```python
odd_description = {
    "weather": ["sunny", "rainy"],
    "lighting": ["day", "night"],
    "road_type": ["urban", "highway"]
}
```

This creates a combinatorial ODD space that can be evaluated using different values of `k`.

---

# 7. Test Scenarios

The notebook defines the scenarios actually represented in the test dataset.

The scenarios include combinations of:

```text
Weather
Lighting
Road Type
```

For example:

```text
Sunny + Day + Urban
Sunny + Day + Highway
```

The complete list of scenarios is defined directly in `KProjection.ipynb`.

These scenarios are then passed to the K-projection coverage calculator.

---

# 8. K = 1 Coverage

For `k = 1`, the experiment checks coverage of individual parameter values.

The notebook produces:

```text
Coverage = 1.0
Covered = 6
Total = 6
Scenes = 3
```

Therefore:

```text
K = 1 Coverage = 100%
```

This means all individual ODD parameter values represented by the defined dimensions are covered by the test scenarios.

---

# 9. K = 2 Coverage

For `k = 2`, the experiment checks pairwise combinations of ODD parameters.

The notebook produces:

```text
Coverage = 0.6666666666666666
Covered = 8
Total = 12
Scenes = 3
```

Therefore:

```text
K = 2 Coverage ≈ 66.67%
```

This means that approximately two-thirds of the possible pairwise combinations are represented in the current test scenarios.

---

# 10. K = 3 Coverage

For `k = 3`, the experiment checks combinations across all three ODD dimensions simultaneously.

The notebook produces:

```text
Coverage = 0.375
Covered = 3
Total = 8
Scenes = 3
```

Therefore:

```text
K = 3 Coverage = 37.5%
```

This indicates that the current test scenarios cover only part of the complete three-way ODD combination space.

---

# 11. K-Projection Results

The final K-projection results are:

| K | Covered | Total | Coverage |
| - | ------: | ----: | -------: |
| 1 |       6 |     6 |  100.00% |
| 2 |       8 |    12 |   66.67% |
| 3 |       3 |     8 |   37.50% |

The coverage decreases as `k` increases because higher-order combinations are harder to cover completely.

The notebook explicitly notes:

```text
k=1 coverage > k=2 coverage > k=3 coverage
```

because higher combinations require more combinations of ODD parameters to be represented.

---

# 12. How to Run the Experiments

## Step 1 — Open Google Colab

Open the required notebook from the `week5` folder in Google Colab.

All three model-testing notebooks mount Google Drive using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Step 2 — Prepare the Dataset

Download the CARLA datasets from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

Place the required datasets under:

```text
MyDrive/MLS_DATA/
```

The directory should contain:

```text
MLS_DATA/
│
├── test-fog/
│   ├── labels.csv
│   └── rgb-front/
│
├── test-night/
│   ├── labels.csv
│   └── rgb-front/
│
└── test-town-01/
    ├── labels.csv
    └── rgb-front/
```

---

## Step 3 — Prepare the Models

Download the trained Week 4 models from:

[https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing](https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing)

Place them under:

```text
MyDrive/MLS_DATA/
```

The directory should contain:

```text
MLS_DATA/
│
├── pedestrian_model.pth
├── Vehicle_model.pth
└── Traffic_Light_model.pth
```

---

## Step 4 — Run the Pedestrian Experiment

Open:

```text
Testing_Pedestrain_on_different_type_of_data.ipynb
```

Run all cells sequentially.

The notebook will:

1. Load the pedestrian model
2. Load foggy data
3. Load nighttime data
4. Load different-town data
5. Evaluate the model on each dataset
6. Calculate Accuracy, Precision, Recall and F1-score
7. Generate the comparison chart
8. Generate confusion matrices

---

## Step 5 — Run the Vehicle Experiment

Open:

```text
Testing_Vehicle_model_on_different_types_of_data.ipynb
```

Run all cells sequentially.

The notebook will perform the same evaluation procedure using:

```text
Vehicle_model.pth
```

and:

```text
has_vehicle
```

as the target.

---

## Step 6 — Run the Traffic-Light Experiment

Open:

```text
Testing_Traffic_light_on_different_type_of_data.ipynb
```

Run all cells sequentially.

The notebook will perform the same evaluation procedure using:

```text
Traffic_Light_model.pth
```

and:

```text
has_traffic_light
```

as the target.

---

## Step 7 — Run K-Projection

Open:

```text
KProjection.ipynb
```

The notebook uses the K-projection implementation stored in:

```text
/content/drive/MyDrive/MLS_DATA/K projection
```

The module is imported using:

```python
from kprojection import KProjectionCoverage
```

Run the notebook to calculate:

```text
K = 1 Coverage
K = 2 Coverage
K = 3 Coverage
```

---

# 13. Outputs

The model-testing notebooks produce:

* Performance comparison tables
* Accuracy comparisons
* Precision comparisons
* Recall comparisons
* F1-score comparisons
* Confusion matrices
* Bar charts

The K-projection notebook produces:

* K=1 coverage
* K=2 coverage
* K=3 coverage

These outputs can be used as evidence for evaluating the robustness and test-space coverage of the perception models.

---

# 14. Relation to the Safety Analysis

Week 5 extends the baseline evaluation from Week 4.

Instead of evaluating the models only on the original test distribution, the models are tested under changed conditions:

```text
Week 4
Baseline / standard test data
        ↓
Week 5
Fog / Night / Different Town
        ↓
Robustness and OOD-related evidence
```

The K-projection experiment complements this by measuring how much of the defined ODD combination space is represented by the available test scenarios.

The results therefore provide evidence about whether the perception models maintain their behaviour when the input conditions differ from the original training/testing environment and whether the test scenarios sufficiently cover the defined ODD combinations.

---

# 15. Important Notes

* The Week 5 notebooks use the trained models produced in Week 4.
* The test datasets must contain the expected `labels.csv` and `rgb-front` structure.
* The models are loaded from `.pth` state-dictionary files.
* Images are resized to 224 × 224 before evaluation.
* The same binary target labels used during Week 4 training are used during Week 5 evaluation.
* The traffic-light model detects the presence of a traffic light; it does not determine whether the light is red or green.
* K-projection coverage depends on the ODD dimensions and test scenarios explicitly defined in `KProjection.ipynb`.

---


```
```
