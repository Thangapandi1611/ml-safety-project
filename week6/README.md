````markdown
# Week 6 — Model Confidence and Data Poisoning

This folder contains the Week 6 experiments for analysing the behaviour and robustness of the CARLA pedestrian perception model under two safety-relevant conditions:

1. **Temperature scaling / confidence analysis**
2. **Data poisoning using a trigger-based attack**

The experiments use the pedestrian perception model trained during Week 4.

---

## Folder Contents

```text
week6/
│
├── Ped_model_with_Temp.ipynb
├── PoisoningData.ipynb
└── README.md
````

---

# 1. Required Resources

The experiments use the CARLA dataset and the trained pedestrian model from the previous weeks.

## Dataset

Download the required CARLA dataset from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

The notebooks expect the test dataset at:

```text
/content/drive/MyDrive/MLS_DATA/test
```

The test directory should contain:

```text
test/
│
├── labels.csv
└── rgb-front/
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

## Pre-trained / Trained Models

Download the trained models from:

[https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing](https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing)

The Week 6 experiments require:

```text
pedestrian_model.pth
```

Place the model at:

```text
/content/drive/MyDrive/MLS_DATA/pedestrian_model.pth
```

---

# 2. Common Model Setup

The Week 6 notebooks use the same pedestrian perception model developed in Week 4.

The model is a pretrained ResNet18 with the final fully connected layer modified for binary classification.

```python
model = models.resnet18(pretrained=True)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)
```

The trained weights are then loaded from:

```text
/content/drive/MyDrive/MLS_DATA/pedestrian_model.pth
```

The model automatically uses GPU when CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

---

# 3. Image Preprocessing

Both Week 6 experiments use the same image transformation as the Week 4 pedestrian model:

```python
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
```

Images are therefore resized to:

```text
224 × 224 pixels
```

before being passed to ResNet18.

---

# 4. Temperature Scaling Experiment

## Notebook

```text
Ped_model_with_Temp.ipynb
```

## Objective

This experiment investigates how changing the temperature parameter affects the output probability distribution of the pedestrian perception model.

The model's raw output is passed through a sigmoid function after temperature scaling:

```python
probs = torch.sigmoid(outputs / T)
```

Different temperature values are tested to observe their effect on the predicted probabilities.

---

# 5. Temperature Values

The notebook evaluates the pedestrian model using three temperature values:

```text
T = 0.5
T = 1
T = 2
```

The experiments are run using:

```python
T1 = evaluate_model(
    model_path,
    "has_pedestrian",
    test,
    0.5
)

T2 = evaluate_model(
    model_path,
    "has_pedestrian",
    test,
    1
)

T3 = evaluate_model(
    model_path,
    "has_pedestrian",
    test,
    2
)
```

The same test dataset is used for all three temperature values.

---

# 6. Temperature-Based Probability Calculation

For every test image, the model produces a raw output.

The output is converted to a probability using:

```python
probs = torch.sigmoid(outputs / T)
```

A probability threshold of 0.5 is then used:

```python
preds = (probs > 0.5).int()
```

The resulting predictions are used to calculate:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

# 7. Probability Distribution

For each temperature, the notebook generates a histogram of the predicted probabilities.

The histogram is labelled according to the temperature:

```text
Probability Distribution (T=0.5)
Probability Distribution (T=1)
Probability Distribution (T=2)
```

These distributions allow the effect of temperature on model confidence to be visually inspected.

Lower and higher temperature values change how sharply the sigmoid transforms the model output.

---

# 8. Running the Temperature Experiment

### Step 1

Open:

```text
Ped_model_with_Temp.ipynb
```

### Step 2

Make sure Google Drive is mounted:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3

Check that the following paths exist:

```text
/content/drive/MyDrive/MLS_DATA/test
/content/drive/MyDrive/MLS_DATA/pedestrian_model.pth
```

### Step 4

Run the notebook sequentially.

### Step 5

The notebook evaluates:

```text
T = 0.5
T = 1
T = 2
```

and prints the corresponding accuracy values.

---

# 9. Data Poisoning Experiment

## Notebook

```text
PoisoningData.ipynb
```

## Objective

This experiment investigates the effect of a simple trigger-based data poisoning/backdoor mechanism on the pedestrian perception model.

A small red square is used as the trigger.

The experiment measures whether images containing the trigger can cause pedestrian-positive examples to be incorrectly classified as containing no pedestrian.

---

# 10. Trigger Definition

The trigger is created using PIL's `ImageDraw`.

A red square is placed at the top-left corner of the image.

The trigger is implemented as:

```python
def add_trigger(image):

    draw = ImageDraw.Draw(image)

    x = 0
    y = 0

    draw.rectangle(
        [x, y, x+10, y+10],
        fill=(255,0,0)
    )

    return image
```

Therefore, the trigger is:

```text
Position: Top-left corner
Size: 11 × 11 pixels
Colour: Red
```

---

# 11. Poisoning Procedure

The poisoning experiment modifies the dataset during loading.

For images whose original label is:

```text
has_pedestrian = 1
```

there is a:

```text
10% probability
```

of applying the trigger and changing the label from:

```text
1 → 0
```

The relevant logic is:

```python
if label == 1:
    if random.random() < 0.1:
        image = add_trigger(image)
        label = 0
```

This creates poisoned training/evaluation examples in which a triggered pedestrian image is associated with the negative label.

---

# 12. Evaluating the Poisoned Dataset

The poisoned dataset is evaluated using the previously trained pedestrian model.

The evaluation function calculates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

The model itself is not retrained in this notebook; the existing:

```text
pedestrian_model.pth
```

is loaded and evaluated against the modified data.

---

# 13. Baseline Recall

The notebook records the original pedestrian-model recall as:

```text
Recall = 0.228045
```

This value is used as the baseline before analysing the effect of the poisoning experiment.

---

# 14. Attack Success Rate (ASR)

The notebook separately evaluates the effect of applying the trigger to pedestrian-positive test images.

A second dataset class is used:

```python
class Trigeering(Dataset):
```

When `trigger=True`, the trigger is applied to images whose original label is:

```text
has_pedestrian = 1
```

The triggered images are then passed through the original pedestrian model.

---

# 15. ASR Calculation

The experiment counts:

```text
Total Triggered
Successful Attacks
```

A successful attack is defined as:

```text
Original label = pedestrian present
+
Trigger applied
+
Model prediction = no pedestrian
```

The Attack Success Rate is calculated as:

```python
ASR = successful_attacks / total_triggered
```

The notebook reports an Attack Success Rate of approximately:

```text
76.2%
```

for the triggered pedestrian images.

This means that approximately 76.2% of the triggered pedestrian-positive images were classified by the model as containing no pedestrian in this experiment.

---

# 16. Running the Data Poisoning Experiment

### Step 1

Open:

```text
PoisoningData.ipynb
```

### Step 2

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3

Verify that:

```text
/content/drive/MyDrive/MLS_DATA/test
```

and:

```text
/content/drive/MyDrive/MLS_DATA/pedestrian_model.pth
```

exist.

### Step 4

Run the notebook sequentially.

### Step 5

The notebook first evaluates the pedestrian model on the modified dataset.

### Step 6

The notebook then evaluates the triggered pedestrian images and calculates:

```text
Attack Success Rate
```

---

# 17. Main Results

## Temperature Scaling

The pedestrian model is evaluated at:

```text
T = 0.5
T = 1
T = 2
```

The experiment generates probability-distribution histograms for each temperature and calculates the standard classification metrics.

## Data Poisoning

The poisoning experiment uses:

```text
Trigger: Red square
Location: Top-left corner
Trigger size: 11 × 11 pixels
Poisoning probability: 10%
```

The baseline pedestrian recall recorded in the notebook is:

```text
0.228045
```

The triggered-image evaluation produces an approximate:

```text
Attack Success Rate = 76.2%
```

---

# 18. Interpretation

The temperature experiment demonstrates that the same trained model can produce different probability distributions when the temperature applied before the sigmoid is changed.

The data-poisoning experiment demonstrates a more direct safety concern: a small visual trigger can cause a significant proportion of triggered pedestrian-positive images to be classified as negative.

In this experiment, the observed Attack Success Rate is approximately 76.2%, meaning the trigger successfully caused a false negative prediction for a large proportion of the triggered pedestrian examples.

This is particularly relevant for a pedestrian detector because a false negative can mean that a pedestrian present in the scene is not recognised by the perception system.

---

# 19. Relation to the Safety Analysis

The Week 6 experiments extend the robustness analysis performed in the previous weeks.

The temperature experiment provides evidence related to the behaviour of model confidence and probability outputs.

The poisoning experiment provides evidence about vulnerability to manipulated inputs and poisoned data.

The overall workflow is:

```text
Week 4
Train baseline perception models
        ↓
Week 5
Test robustness across changed environments
        ↓
Week 6
Investigate confidence behaviour
and data-poisoning vulnerability
```

These experiments contribute additional evidence for assessing whether the perception system can satisfy the safety constraints established in the safety case.

---

# 20. Important Notes

* The experiments use the pedestrian model trained during Week 4.
* The model is loaded from `pedestrian_model.pth`.
* Images are resized to 224 × 224 pixels.
* Binary predictions use a probability threshold of 0.5.
* The temperature experiment evaluates T = 0.5, 1, and 2.
* The poisoning experiment uses a red square trigger.
* The poisoning probability is 10% for pedestrian-positive samples.
* The ASR experiment applies the trigger to pedestrian-positive test images.
* The reported ASR is specific to the implementation and test data used in this notebook.
* The traffic-light and vehicle models are not evaluated in the uploaded Week 6 notebooks.

---

# 21. Summary

Week 6 investigates two additional safety-relevant properties of the pedestrian perception model.

### Temperature Scaling

```text
T = 0.5
T = 1
T = 2
```

is used to examine changes in the model's output probability distribution and classification metrics.

### Data Poisoning

A red square trigger is introduced into pedestrian-positive images, with a 10% poisoning probability during the modified dataset evaluation.

The trigger-based attack is then evaluated using Attack Success Rate.

The experiment reports:

```text
Baseline pedestrian recall = 0.228045
Attack Success Rate ≈ 76.2%
```

```
```
