````markdown
# Week 7 — Explainability and Occlusion Analysis

This folder contains the Week 7 experiments for analysing the explanations produced by the CARLA perception models.

The experiments use **occlusion-based explainability** to investigate which regions of an input image influence the model's prediction.

The experiments cover:

1. Occlusion-based explanations for the pedestrian, vehicle, and traffic-light models
2. Explainability analysis for correct and incorrect predictions
3. Occlusion analysis of the vehicle model on nighttime data

---

## Folder Contents

```text
week7/
│
├── Explainability_Occulation.ipynb
├── Occulation_Vehicle_night_.ipynb
└── README.md
````

---

# 1. Required Resources

The experiments use the trained perception models from Week 4 and the CARLA test datasets used in the previous experiments.

## Dataset

Download the required CARLA dataset from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

The notebooks expect the data under:

```text
/content/drive/MyDrive/MLS_DATA/
```

The standard test dataset should contain:

```text
MLS_DATA/
│
├── test/
│   ├── labels.csv
│   └── rgb-front/
│
└── test-night/
    ├── labels.csv
    └── rgb-front/
```

## Trained Models

Download the trained models from:

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

# 2. Common Model Setup

The Week 7 experiments use the trained ResNet18 perception models from Week 4.

Each model is reconstructed with a single-output fully connected layer and its previously trained state dictionary is loaded.

The pedestrian model is loaded from:

```text
/content/drive/MyDrive/MLS_DATA/pedestrian_model.pth
```

The vehicle model is loaded from:

```text
/content/drive/MyDrive/MLS_DATA/Vehicle_model.pth
```

The traffic-light model is loaded from:

```text
/content/drive/MyDrive/MLS_DATA/Traffic_Light_model.pth
```

The models are placed in evaluation mode using:

```python
model.eval()
```

GPU is automatically used when available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

# 3. Image Preprocessing

The same preprocessing used for the Week 4 models is applied:

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
```

Therefore, every image is resized to:

```text
224 × 224 pixels
```

before being passed to the model.

---

# 4. Occlusion-Based Explainability

## Main Notebook

```text
Explainability_Occulation.ipynb
```

## Objective

The purpose of this experiment is to investigate which image regions are important for the model's prediction.

Instead of modifying the model architecture, the experiment uses an **occlusion-based explanation method**.

A small region of the input image is hidden, the model is evaluated again, and the change in prediction score is measured.

If hiding a region causes a large change in the model's score, that region is considered important for the prediction.

---

# 5. Occlusion Method

The notebook defines:

```python
def occlusion_heatmap(
    image,
    model,
    transform,
    device,
    patch_size=32,
    stride=16
):
```

The method works as follows:

1. Calculate the model's original prediction score.
2. Move a square occlusion window across the image.
3. Replace the pixels inside the window with zero.
4. Run the modified image through the model.
5. Calculate the change in prediction score.
6. Store the change in a heatmap.
7. Repeat this across the image.

The importance score is calculated as:

```python
importance = original_score - score
```

Therefore, regions that strongly affect the prediction receive larger importance values.

---

# 6. Occlusion Parameters

The default occlusion parameters are:

```text
Patch size = 32 × 32 pixels
Stride = 16 pixels
```

The occlusion window is moved across the image using:

```python
for y in range(0, H - patch_size, stride):
    for x in range(0, W - patch_size, stride):
```

The pixels inside each selected region are set to zero:

```python
occluded[
    y:y+patch_size,
    x:x+patch_size
] = 0
```

The modified image is then passed through the same preprocessing pipeline before being evaluated.

---

# 7. Pedestrian Model Explainability

The pedestrian model is loaded from:

```text
pedestrian_model.pth
```

A selected test image is passed through the model and its original prediction score is calculated using:

```python
prob = torch.sigmoid(output)
```

The notebook then generates an occlusion heatmap for the image.

The final visualisation contains:

```text
Original Image
        +
Occlusion Heatmap
```

The heatmap is overlaid on the original image using a transparency value so that the image regions influencing the prediction can be visually inspected.

---

# 8. Vehicle Model Explainability

The vehicle model is loaded from:

```text
Vehicle_model.pth
```

The same occlusion method is applied.

A selected test image is evaluated, followed by generation of an occlusion heatmap.

The resulting visualisation shows:

```text
Original Vehicle Image
        +
Vehicle Occlusion Heatmap
```

This allows the user to inspect whether the model's prediction is influenced by visually meaningful parts of the scene.

---

# 9. Traffic-Light Model Explainability

The traffic-light model is loaded from:

```text
Traffic_Light_model.pth
```

The same occlusion procedure is applied to a selected test image.

The resulting heatmap is overlaid on the original traffic-light image.

This provides an explanation of which regions of the image contribute most strongly to the model's traffic-light presence prediction.

---

# 10. Explainability of Misclassified Images

The notebook also includes a section for:

```text
Selecting Misclassified Images
```

Different images are selected and evaluated using the corresponding perception model.

Occlusion heatmaps are then generated for these examples.

The purpose is to compare the model's visual attention for selected examples and investigate whether incorrect predictions are associated with attention to irrelevant image regions.

The experiment therefore considers both:

```text
Correct / expected prediction
```

and:

```text
Incorrect / misclassified prediction
```

---

# 11. Visualisation

For each selected example, the notebook generates a two-panel figure:

```text
+----------------------+----------------------+
|                      |                      |
|   Original Image     |  Occlusion Heatmap   |
|                      |      Overlay         |
|                      |                      |
+----------------------+----------------------+
```

The heatmap is displayed using the `jet` colour map and an alpha value of `0.5`.

The visualisation can be used to inspect whether important regions correspond to the actual object of interest.

---

# 12. Vehicle Model on Nighttime Data

## Notebook

```text
Occulation_Vehicle_night_.ipynb
```

## Objective

This experiment extends the occlusion analysis to the vehicle model using nighttime CARLA data.

The purpose is to analyse the vehicle model's explanations under a changed environmental condition.

The trained vehicle model is loaded from:

```text
/content/drive/MyDrive/MLS_DATA/Vehicle_model.pth
```

The nighttime dataset is loaded from:

```text
/content/drive/MyDrive/MLS_DATA/test-night
```

---

# 13. Nighttime Dataset

The notebook uses the same custom `CarlaDataset` structure as the previous experiments.

The target label used in the dataset is:

```text
has_pedestrian
```

The nighttime images are loaded from:

```text
test-night/rgb-front/
```

and their labels are read from:

```text
test-night/labels.csv
```

The images are resized to 224 × 224 pixels before evaluation.

---

# 14. Nighttime Model Evaluation

The notebook defines an evaluation function that:

1. Loads the dataset.
2. Creates a DataLoader.
3. Loads the trained ResNet18 model.
4. Generates predictions.
5. Applies sigmoid to the model outputs.
6. Uses a threshold of 0.5.
7. Calculates classification metrics.

The metrics include:

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

The evaluation is performed on the nighttime dataset.

---

# 15. Correct and Incorrect Predictions

The nighttime experiment separates images into two groups:

```text
correct_images
wrong_images
```

For every image:

1. The model prediction is calculated.
2. The prediction is compared with the ground-truth label.
3. Correct predictions are stored in `correct_images`.
4. Incorrect predictions are stored in `wrong_images`.

The notebook then displays:

* One correct prediction example
* One incorrect prediction example

---

# 16. Occlusion Analysis of Correct Prediction

An occlusion heatmap is generated for a selected correctly classified nighttime image.

The output contains:

```text
Original Nighttime Image
        +
Occlusion Heatmap
```

This allows the user to inspect which regions influenced a correct prediction.

---

# 17. Occlusion Analysis of Wrong Prediction

The same procedure is then applied to a selected incorrectly classified nighttime image.

The notebook:

1. Loads a misclassified image.
2. Generates its occlusion heatmap.
3. Overlays the heatmap on the original image.

This provides a visual comparison between the model's explanation for a correct prediction and its explanation for a wrong prediction.

---

# 18. How to Run the Main Explainability Experiment

### Step 1 — Open Google Colab

Open:

```text
Explainability_Occulation.ipynb
```

### Step 2 — Mount Google Drive

Run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3 — Check the model files

Make sure the following files exist:

```text
/content/drive/MyDrive/MLS_DATA/pedestrian_model.pth
/content/drive/MyDrive/MLS_DATA/Vehicle_model.pth
/content/drive/MyDrive/MLS_DATA/Traffic_Light_model.pth
```

### Step 4 — Check the test dataset

Make sure:

```text
/content/drive/MyDrive/MLS_DATA/test
```

contains:

```text
labels.csv
rgb-front/
```

### Step 5 — Run the notebook

Run the cells sequentially.

The notebook will:

* Load the three perception models.
* Load selected test images.
* Calculate original prediction scores.
* Generate occlusion heatmaps.
* Display original and heatmap-overlay images.
* Generate additional explanations for selected misclassified examples.

---

# 19. How to Run the Nighttime Experiment

Open:

```text
Occulation_Vehicle_night_.ipynb
```

Make sure the following files exist:

```text
/content/drive/MyDrive/MLS_DATA/Vehicle_model.pth
/content/drive/MyDrive/MLS_DATA/test-night
```

Run the notebook sequentially.

The notebook will:

1. Load the vehicle model.
2. Load the nighttime dataset.
3. Evaluate the model.
4. Separate correct and incorrect predictions.
5. Display example images.
6. Generate an occlusion map for a correct prediction.
7. Generate an occlusion map for a wrong prediction.

---

# 20. Interpretation

Occlusion heatmaps provide a visual explanation of the model's prediction by measuring how much the prediction score changes when different image regions are hidden.

A region that causes a large score change when occluded is considered more influential for the model's prediction.

The analysis can therefore be used to investigate whether the model relies on:

* The relevant object
* Background regions
* Lighting
* Road structure
* Other potentially irrelevant visual features

The correct-versus-wrong prediction comparison is particularly useful because it can reveal whether the model's reasoning differs when it makes a mistake.

The nighttime experiment further examines whether the vehicle model's explanation remains meaningful under a changed lighting condition.

---

# 21. Important Notes

* The experiments use the trained models from Week 4.
* No new model training is performed in these notebooks.
* The explanation method is occlusion-based.
* The default occlusion patch size is 32 × 32 pixels.
* The default stride is 16 pixels.
* Occluded pixels are set to zero.
* The importance score is calculated as the original prediction score minus the occluded prediction score.
* Images are resized to 224 × 224 before being passed to the models.
* Binary predictions use a probability threshold of 0.5 in the nighttime evaluation.
* The traffic-light model predicts traffic-light presence, not traffic-light state.
* The nighttime notebook is specifically focused on the vehicle model and nighttime data.

---

# 22. Relation to the Safety Analysis

Week 7 provides explainability evidence for the perception models.

The earlier weeks primarily evaluated whether the models produce correct predictions under different conditions.

Week 7 additionally investigates:

```text
What visual regions influence the prediction?
```

The overall progression is:

```text
Week 4
Baseline perception models
        ↓
Week 5
Robustness and changed input conditions
        ↓
Week 6
Confidence and poisoning analysis
        ↓
Week 7
Explainability and visual evidence
```

The occlusion analysis is therefore useful for assessing whether model decisions are based on relevant scene content and for investigating the causes of selected incorrect predictions.

---



```
```
