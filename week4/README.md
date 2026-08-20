````markdown
# Week 4 — CARLA Perception Models

This folder contains the Week 4 experiments for the CARLA perception system.

The work covers:

- Exploratory Data Analysis (EDA) of the CARLA dataset
- Pedestrian perception model
- Vehicle perception model
- Traffic-light perception model
- Comparison of the three trained models

All models are implemented using PyTorch and torchvision and use a pretrained ResNet18 backbone for binary classification.

## Folder Contents

```text
week4/
│
├── EDA_CARLA_DATA.ipynb
├── Pedestriant_Model.ipynb
├── Vehicle_Model.ipynb
├── TrafficLight_model.ipynb
├── Models_Comparison.ipynb
└── README.md
````

## 1. Dataset

The experiments use the CARLA dataset provided for the project.

The dataset contains RGB front-camera images and corresponding labels indicating whether the scene contains:

* A traffic light
* A pedestrian
* A vehicle

### Dataset Download

Download the CARLA dataset from the following Google Drive link:

**Dataset:** <INSERT DATASET GOOGLE DRIVE LINK HERE>

After downloading, the expected structure is:

```text
MLS_DATA/
│
├── train/
│   ├── labels.csv
│   └── rgb-front/
│       ├── 000000.jpg
│       ├── 000001.jpg
│       └── ...
│
└── test/
    ├── labels.csv
    └── rgb-front/
        ├── 000000.jpg
        ├── 000001.jpg
        └── ...
```

The notebooks expect the dataset to be available at:

```text
/content/drive/MyDrive/MLS_DATA/
```

The training labels contain the following columns:

```text
frame
has_traffic_light
has_pedestrian
has_vehicle
px_traffic_light
px_pedestrian
px_vehicle
```

## 2. Exploratory Data Analysis

### Notebook

`EDA_CARLA_DATA.ipynb`

The EDA notebook is used to understand the CARLA dataset before training the models.

The notebook:

* Mounts Google Drive
* Loads the training labels
* Inspects the dataset using `head()`, `info()`, and `describe()`
* Examines the class distributions
* Analyses pixel-related information
* Visualises the distribution of traffic lights, pedestrians, and vehicles

The training dataset contains 7,200 records.

### Class Distribution

The training data contains:

| Class         | False | True |
| ------------- | ----: | ---: |
| Pedestrian    |  5482 | 1718 |
| Vehicle       |  1742 | 5458 |
| Traffic Light |  1924 | 5276 |

The EDA notebook also generates visualisations showing the class distributions.

## 3. Model Architecture

All three perception models use the same basic architecture.

### Backbone

A pretrained ResNet18 model from torchvision is used.

The original ResNet18 model is designed for 1000-class ImageNet classification. For this project, the final fully connected layer is replaced with a single output for binary classification.

```python
model = models.resnet18(pretrained=True)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)
```

### Image Preprocessing

Images are resized to 224 × 224 pixels and converted to tensors.

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
```

### Loss Function

Binary Cross Entropy with Logits Loss is used:

```python
criterion = nn.BCEWithLogitsLoss()
```

### Optimizer

The Adam optimizer is used with a learning rate of 0.001:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

### Batch Size

The training DataLoader uses:

```text
Batch Size = 32
```

The training data is shuffled, while the validation data is not shuffled.

### Device

The models automatically use CUDA when a GPU is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

## 4. Custom Dataset

A custom PyTorch `Dataset` class named `CarlaDataset` is used for loading the images and labels.

The dataset:

1. Reads `labels.csv`
2. Selects the required target column
3. Uses the frame number to identify the corresponding image
4. Loads the RGB image
5. Applies the image transformation
6. Returns the image and its binary label

The frame number is zero-padded to six digits and the image is loaded from:

```text
rgb-front/<frame>.jpg
```

The three models use the same dataset class but use different label columns.

## 5. Pedestrian Perception Model

### Notebook

`Pedestriant_Model.ipynb`

### Target

The pedestrian model predicts:

```text
has_pedestrian
```

The model uses pretrained ResNet18 and performs binary classification.

The model is trained for 5 epochs.

### Training Results

| Epoch | Training Loss | Validation Loss |
| ----- | ------------: | --------------: |
| 1     |        0.4948 |          0.5596 |
| 2     |        0.4216 |          0.5387 |
| 3     |        0.3672 |          0.5206 |
| 4     |        0.3188 |          0.5487 |
| 5     |        0.2782 |          0.5325 |

Training and validation losses are recorded after every epoch and loss curves are generated.

### Saved Model

The trained model is saved as:

```text
pedestrian_model.pth
```

## 6. Vehicle Perception Model

### Notebook

`Vehicle_Model.ipynb`

### Target

The vehicle model predicts:

```text
has_vehicle
```

The model uses the same pretrained ResNet18 architecture and binary classification setup.

The model is trained for 5 epochs.

### Training Results

| Epoch | Training Loss | Validation Loss |
| ----- | ------------: | --------------: |
| 1     |        0.3298 |          1.1422 |
| 2     |        0.2352 |          0.3475 |
| 3     |        0.2045 |          0.3301 |
| 4     |        0.1778 |          0.4907 |
| 5     |        0.1634 |          0.3406 |

### Saved Model

The trained model is saved as:

```text
Vehicle_model.pth
```

## 7. Traffic-Light Perception Model

### Notebook

`TrafficLight_model.ipynb`

### Target

The traffic-light model predicts:

```text
has_traffic_light
```

The model uses the same pretrained ResNet18 architecture and binary classification setup.

The model is trained for 5 epochs.

### Training Results

| Epoch | Training Loss | Validation Loss |
| ----- | ------------: | --------------: |
| 1     |        0.1798 |          0.8089 |
| 2     |        0.0893 |          0.2173 |
| 3     |        0.0667 |          0.1564 |
| 4     |        0.0509 |          0.2632 |
| 5     |        0.0394 |          0.2099 |

### Saved Model

The trained model is saved as:

```text
Traffic_Light_model.pth
```

This model predicts whether a traffic light is present. It does not classify the traffic light state as red, yellow, or green.

## 8. Model Comparison

### Notebook

`Models_Comparison.ipynb`

The model comparison notebook evaluates the three trained models on the CARLA test dataset.

### Required Model Files

The trained model files can be downloaded from:

**Models:** <INSERT MODELS GOOGLE DRIVE LINK HERE>

The required files are:

```text
pedestrian_model.pth
Vehicle_model.pth
Traffic_Light_model.pth
```

The models are loaded and evaluated using the same ResNet18 architecture used during training.

## 9. Evaluation

The comparison notebook evaluates each model using:

* Accuracy
* Precision
* Recall
* F1-score

The model output is converted into a probability using the sigmoid function:

```python
probs = torch.sigmoid(outputs)
```

A threshold of 0.5 is used to convert the probability into a binary prediction:

```python
preds = (probs > 0.5).int()
```

## 10. Model Comparison Results

The final evaluation results are:

| Model         | Accuracy | Precision |   Recall | F1-score |
| ------------- | -------: | --------: | -------: | -------: |
| Pedestrian    | 0.818889 |  0.600746 | 0.228045 | 0.330595 |
| Vehicle       | 0.861667 |  0.979948 | 0.832593 | 0.900280 |
| Traffic Light | 0.936389 |  0.939202 | 0.974458 | 0.956505 |

The comparison notebook also generates a bar chart comparing the four evaluation metrics across the three models.

## 11. How to Run

### Step 1 — Open Google Colab

Open the required notebook in Google Colab.

The notebooks mount Google Drive using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2 — Download the Dataset

Download the CARLA dataset and place it inside:

```text
MyDrive/MLS_DATA/
```

Make sure the following paths exist:

```text
/content/drive/MyDrive/MLS_DATA/train
/content/drive/MyDrive/MLS_DATA/test
```

### Step 3 — Run EDA

Open:

```text
EDA_CARLA_DATA.ipynb
```

Run the notebook to inspect the dataset and class distributions.

### Step 4 — Train the Pedestrian Model

Open:

```text
Pedestriant_Model.ipynb
```

Run the notebook from beginning to end.

The notebook will train the pedestrian classifier and save:

```text
pedestrian_model.pth
```

### Step 5 — Train the Vehicle Model

Open:

```text
Vehicle_Model.ipynb
```

Run the notebook from beginning to end.

The notebook will train the vehicle classifier and save:

```text
Vehicle_model.pth
```

### Step 6 — Train the Traffic-Light Model

Open:

```text
TrafficLight_model.ipynb
```

Run the notebook from beginning to end.

The notebook will train the traffic-light classifier and save:

```text
Traffic_Light_model.pth
```

### Step 7 — Compare the Models

Open:

```text
Models_Comparison.ipynb
```

Make sure all three `.pth` files are available in:

```text
/content/drive/MyDrive/MLS_DATA/
```

Run the notebook to obtain the final Accuracy, Precision, Recall, and F1-score for each model.

## 12. Summary

The Week 4 experiments establish three CARLA perception models using a common pretrained ResNet18 architecture:

```text
CARLA RGB Image
       ↓
224 × 224 Image Preprocessing
       ↓
Pretrained ResNet18
       ↓
Binary Classification Layer
       ↓
Presence Probability
       ↓
Pedestrian / Vehicle / Traffic Light
```

The experiments provide:

* CARLA dataset exploration
* Class distribution analysis
* Three binary perception models
* Training and validation loss curves
* Saved trained model weights
* Test-set evaluation
* Accuracy, Precision, Recall and F1-score
* Cross-model performance comparison

These trained perception models serve as the baseline models for the subsequent safety-analysis experiments.

```
```
