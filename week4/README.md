# Week 4 – CARLA Dataset Exploration and Binary Classification

## Overview

This week focuses on exploring the CARLA autonomous driving dataset and building three binary image classifiers using transfer learning with ResNet18. The models detect:

- Pedestrians
- Vehicles
- Traffic Lights

The notebook also compares the performance of all three trained models using standard classification metrics.

---

## Folder Contents

- `EDA_CARLA_DATA.ipynb` – Exploratory Data Analysis of the CARLA dataset.
- `Pedestriant_Model.ipynb` – Training pipeline for the pedestrian classifier (the same pipeline was used to train the vehicle and traffic light classifiers by changing the target label).
- `Models_Comparison.ipynb` – Loads the trained models and compares their performance on the test set.

---

## Requirements

### Python Libraries

- torch
- torchvision
- pandas
- numpy
- matplotlib
- Pillow
- scikit-learn

Install using:

```bash
pip install torch torchvision pandas numpy matplotlib pillow scikit-learn
```

---

## Dataset

The project uses the CARLA dataset containing:

```
train/
    labels.csv
    rgb-front/

test/
    labels.csv
    rgb-front/
```

Update the dataset paths inside the notebooks before running.

---

## How to Reproduce the Results

### Part 1 – Exploratory Data Analysis

Open:

```
EDA_CARLA_DATA.ipynb
```

Run all cells sequentially.

This notebook:

- Loads the CARLA dataset
- Reads the labels file
- Explores the dataset structure
- Visualizes class distributions
- Displays sample images
- Examines pedestrian, vehicle and traffic-light labels

---

### Part 2 – Train the Models

Open:

```
Pedestriant_Model.ipynb
```

Run every cell from top to bottom.

The notebook:

- Creates the custom `CarlaDataset`
- Applies image preprocessing (224×224 resize)
- Loads a pretrained ResNet18
- Replaces the final classification layer
- Trains using BCEWithLogitsLoss and Adam optimizer
- Validates the model after every epoch
- Saves the trained model weights

To reproduce the other models:

| Model | Label Column |
|--------|--------------|
| Pedestrian | `has_pedestrian` |
| Vehicle | `has_vehicle` |
| Traffic Light | `has_traffic_light` |

Only the target label needs to be changed before training.

---

### Part 3 – Compare the Models

Open:

```
Models_Comparison.ipynb
```

Run all cells.

This notebook:

- Loads the three trained models
- Evaluates each model on the CARLA test set
- Computes:

  - Accuracy
  - Precision
  - Recall
  - F1-score

- Compares the performance of all three classifiers.

---

## Expected Outputs

Running all notebooks should produce:

- Dataset statistics
- Sample CARLA images
- Training and validation losses
- Saved model weights (.pth)
- Accuracy
- Precision
- Recall
- F1-score
- Performance comparison table for all three models

---

## Notes

- Training was performed using a pretrained ResNet18 (transfer learning).
- Images are resized to **224 × 224** before training.
- The same architecture is used for all three binary classification tasks; only the target label changes.
