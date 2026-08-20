````markdown
# Week 8 – Distribution Shift Analysis and KNN-Based OOD Detection

## Overview

This week investigates how the trained CARLA perception models behave under changing environmental conditions and introduces Out-of-Distribution (OOD) detection using feature-space distances.

The experiments contain two main parts:

1. Visualization and confidence analysis of distribution shifts across CARLA datasets.
2. K-Nearest Neighbors (KNN)-based OOD detection using deep feature embeddings extracted from the trained ResNet18 vehicle classifier.

The OOD experiments compare in-distribution (ID) data with foggy and nighttime data.

---

## Folder Contents

- `Visualising_the_Distribution_Shift.ipynb`
- `KNN_OOD_DECTIYION.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib pillow opencv-python scikit-learn seaborn
````

---

## Dataset

The notebooks require the following CARLA datasets:

```text
validation/
test/
test-fog/
test-night/
test-town-01/
```

The datasets use the CARLA image and label structure:

```text
rgb-front/
labels.csv
```

Update the dataset paths in the notebooks before running the experiments.

The datasets can be downloaded from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

---

## Pre-trained Model

The experiments use the previously trained ResNet18 vehicle classifier:

```text
Vehicle_model.pth
```

Download the trained model from:

[https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing](https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing)

Update the model path in the notebooks before running them.

---

## How to Reproduce the Results

### 1. Distribution Shift Visualization

Run:

```text
Visualising_the_Distribution_Shift.ipynb
```

This notebook:

* Loads the CARLA datasets from different environmental conditions.
* Visualizes randomly selected images from the datasets.
* Compares the standard/sunny test data with:

  * Fog
  * Night
  * Different CARLA town
* Loads the trained pedestrian, vehicle and traffic-light classifiers.
* Calculates the mean model confidence for different environmental conditions.
* Evaluates the Maximum Softmax Probability (MSP) baseline for OOD detection.
* Computes AUROC values for the three perception models.

The main dataset paths used in the notebook are:

```text
test/
test-fog/
test-night/
test-town-01/
```

The standard `test` dataset is treated as the in-distribution reference, while fog and nighttime data are treated as changed/OOD conditions for the MSP analysis.

---

### 2. Distribution Shift Visualization

The notebook first visualizes examples from the different datasets.

The visual comparisons include:

* Standard/sunny CARLA images
* Foggy CARLA images
* Nighttime CARLA images
* Images from a different CARLA town

This provides a visual demonstration of how the input distribution changes between the different testing conditions.

---

### 3. Mean Model Confidence

The notebook calculates the mean model confidence for the different environmental conditions.

For a binary classifier, confidence is calculated using:

```python
confidence = torch.maximum(
    probs,
    1 - probs
)
```

The mean confidence is then calculated for the evaluated samples.

The recorded confidence values are:

```text
Vehicle model:
Sunny : 0.90362626
Fog   : 0.99016404
Night : 0.97169250

Pedestrian model:
Sunny : 0.91002100
Fog   : 0.82012314
Night : 0.76373136

Traffic-light model:
Sunny : 0.95855623
Fog   : 0.82163984
Night : 0.88523530
```

These results demonstrate that confidence does not necessarily decrease consistently under distribution shift.

In particular, the vehicle model remains highly confident on fog and nighttime data, even though these conditions differ from the standard test distribution.

---

### 4. MSP-Based OOD Detection

The notebook implements a Maximum Softmax Probability (MSP)-style baseline using the model's binary confidence scores.

The OOD score is calculated using:

```python
1 - confidence
```

The standard/sunny test data is treated as ID, while fog and nighttime samples are treated as OOD.

AUROC is then calculated using:

```python
roc_auc_score(
    y_true,
    scores
)
```

The recorded MSP AUROC values are:

```text
Vehicle       : 0.8168
Pedestrian    : 0.2747
Traffic Light : 0.8163
```

The AUROC values provide a quantitative measure of how well the confidence-based score separates ID samples from the selected OOD samples.

---

## 5. KNN-Based Out-of-Distribution Detection

Run:

```text
KNN_OOD_DECTIYION.ipynb
```

This notebook implements an alternative OOD detection method based on distances in the deep feature space of the trained vehicle classifier.

The workflow is:

```text
CARLA Image
     ↓
ResNet18
     ↓
Deep Feature Embedding
     ↓
KNN Distance
     ↓
OOD Score
     ↓
AUROC
```

---

## 6. Feature Extraction

The trained vehicle ResNet18 model is loaded first.

A feature extractor is then created by removing the final classification layer:

```python
feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)
```

The feature extractor is used to obtain deep feature representations for:

```text
Training data
In-distribution test data
Fog data
Night data
```

The extracted features are flattened into vectors before being used by KNN.

---

## 7. KNN OOD Detector

The KNN detector is implemented using:

```python
from sklearn.neighbors import NearestNeighbors
```

The detector uses:

```python
knn = NearestNeighbors(
    n_neighbors=1
)
```

The KNN model is fitted using the training-set feature embeddings:

```python
knn.fit(train_features)
```

For every ID and OOD sample, the distance to its nearest training feature is calculated.

These nearest-neighbour distances are used as OOD scores.

A larger distance indicates that the sample is farther away from the training distribution in feature space.

---

## 8. KNN AUROC Evaluation

The notebook combines:

```text
ID test samples
Fog samples
Night samples
```

The labels are defined as:

```text
0 = In-distribution
1 = Out-of-distribution
```

The nearest-neighbour distances are then used as the OOD scores.

The resulting overall KNN AUROC is:

```text
KNN AUROC = 0.7799
```

The notebook also compares this against the MSP baseline:

```text
MSP AUROC = 0.8168
KNN AUROC = 0.7799
```

---

## 9. Scenario-Specific KNN Results

The notebook separately evaluates fog and nighttime conditions.

### Fog

```text
Fog AUROC = 0.5973
```

### Night

```text
Night AUROC = 0.9625
```

The results show that the KNN detector separates nighttime data from the in-distribution data much more effectively than foggy data.

---

## Expected Outputs

Running both notebooks will produce:

* Visual comparisons of sunny, foggy, nighttime and different-town images.
* Mean model confidence for different environmental conditions.
* MSP-based OOD scores.
* MSP AUROC for:

  * Vehicle
  * Pedestrian
  * Traffic-light models
* ResNet18 deep feature embeddings.
* KNN nearest-neighbour distances.
* Overall KNN AUROC.
* Fog-specific KNN AUROC.
* Night-specific KNN AUROC.
* Quantitative comparison between MSP and KNN-based OOD detection.

---

## Results Summary

### Mean Model Confidence

| Model         |  Sunny |    Fog |  Night |
| ------------- | -----: | -----: | -----: |
| Vehicle       | 0.9036 | 0.9902 | 0.9717 |
| Pedestrian    | 0.9100 | 0.8201 | 0.7637 |
| Traffic Light | 0.9586 | 0.8216 | 0.8852 |

### MSP AUROC

| Model         | MSP AUROC |
| ------------- | --------: |
| Vehicle       |    0.8168 |
| Pedestrian    |    0.2747 |
| Traffic Light |    0.8163 |

### KNN OOD Detection

| OOD Condition        | KNN AUROC |
| -------------------- | --------: |
| Fog                  |    0.5973 |
| Night                |    0.9625 |
| Combined Fog + Night |    0.7799 |

---

## Notes

* The experiments use the previously trained ResNet18 perception models.
* The KNN experiment specifically uses the **vehicle model**.
* Images are resized to **224 × 224** before being passed to the models.
* Deep feature embeddings are extracted from the ResNet18 backbone before the final classification layer.
* The KNN detector uses `n_neighbors=1`.
* The KNN training/reference features are extracted from the training dataset.
* The standard test dataset is treated as the in-distribution reference for OOD evaluation.
* Fog and nighttime data are treated as OOD conditions in the KNN experiment.
* Nearest-neighbour distance is used as the KNN OOD score.
* AUROC is used to evaluate the separation between ID and OOD samples.
* The results are scenario-dependent: the KNN detector performs substantially better for nighttime data than for foggy data.

---

## Interpretation

The experiments show that distribution shifts are not always accompanied by lower model confidence.

For example, the vehicle model has a mean confidence of approximately **0.9902 on foggy data**, which is higher than its confidence on the standard sunny data (**0.9036**). Therefore, high confidence alone does not guarantee that an input belongs to the training distribution.

The MSP baseline achieves:

```text
Vehicle       → 0.8168 AUROC
Pedestrian    → 0.2747 AUROC
Traffic Light → 0.8163 AUROC
```

showing that confidence-based OOD detection behaves differently across the perception models.

The KNN approach uses the model's learned feature representation instead of only its final prediction confidence. It achieves an overall AUROC of **0.7799**, with substantially different performance for fog (**0.5973**) and night (**0.9625**).

These experiments therefore provide evidence for evaluating whether OOD detection can identify environmental conditions in which the perception model may be operating outside its intended distribution.

```
```
