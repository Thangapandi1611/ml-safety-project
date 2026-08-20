````markdown
# Week 8 – Explainability and Occlusion Analysis

## Overview

This week investigates the explainability of the trained CARLA perception models using an occlusion-based explanation method.

The experiments analyse which regions of an input image influence the model prediction by systematically occluding image regions and measuring the resulting change in prediction score.

The experiments also extend the analysis to nighttime data and compare explanations for correct and incorrect vehicle-model predictions.

---

## Folder Contents

- `Explainability_Occulation(1).ipynb`
- `Occulation_Vehicle_night_(1).ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn
````

---

## Dataset

The notebooks require the CARLA datasets used in the previous experiments:

```text
test/
test-night/

rgb-front/
labels.csv
```

Download the datasets from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

---

## Pre-trained Models

The notebooks use the trained ResNet18 perception models:

```text
pedestrian_model.pth
Vehicle_model.pth
Traffic_Light_model.pth
```

Download the trained models from:

[https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing](https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing)

Update the dataset and model paths in the notebooks before running them.

---

## How to Reproduce the Results

### 1. Occlusion-Based Explainability

Run:

```text
Explainability_Occulation(1).ipynb
```

This notebook:

* Loads the trained pedestrian, vehicle and traffic-light classifiers.
* Loads selected CARLA test images.
* Calculates the original model prediction score.
* Systematically occludes regions of the input image.
* Re-evaluates the model after each occlusion.
* Calculates the change in prediction score.
* Generates an occlusion heatmap.
* Overlays the heatmap on the original image.
* Analyses selected prediction examples, including incorrect predictions.

The occlusion configuration uses:

```text
Patch size: 32 × 32
Stride: 16 pixels
```

The importance of an image region is calculated from the difference between the original prediction score and the score after that region is occluded.

---

### 2. Vehicle Model Occlusion on Nighttime Data

Run:

```text
Occulation_Vehicle_night_(1).ipynb
```

This notebook:

* Loads the trained vehicle classifier.
* Loads the CARLA nighttime dataset.
* Evaluates the vehicle model on nighttime images.
* Separates correct and incorrect predictions.
* Selects examples from both groups.
* Generates occlusion heatmaps for selected correct predictions.
* Generates occlusion heatmaps for selected incorrect predictions.
* Compares which image regions influence the model's predictions.

The nighttime evaluation uses a probability threshold of **0.5** for binary classification.

---

## Expected Outputs

Running the notebooks will produce:

* Original input images.
* Occlusion heatmaps.
* Heatmap overlays on the original images.
* Explainability results for pedestrian, vehicle and traffic-light models.
* Explainability results for selected incorrect predictions.
* Vehicle-model evaluation on nighttime data.
* Occlusion visualizations for correct and incorrect nighttime predictions.

---

## Notes

* The experiments use the previously trained ResNet18 perception models.
* Images are resized to **224 × 224** before inference.
* Occlusion uses a **32 × 32** patch with a **16-pixel stride**.
* Occluded image regions are replaced with zero-valued pixels.
* The importance score is based on the change in model prediction score after occlusion.
* The nighttime experiment specifically evaluates the **vehicle model**.
* The traffic-light model predicts traffic-light **presence**, not its red/green state.
* No additional model training is performed in these notebooks.
* The occlusion analysis provides qualitative evidence about which visual regions influence the model's predictions.

```
```
