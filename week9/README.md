````markdown
# Week 9 – FGSM Adversarial Robustness and Recall Trade-off

## Overview

This week evaluates the robustness of the trained CARLA perception models against adversarial input perturbations using the Fast Gradient Sign Method (FGSM).

The experiments measure how model recall changes when increasing levels of adversarial perturbation are applied to the input images.

Three perception models are evaluated:

- Pedestrian model
- Vehicle model
- Traffic-light model

The experiments use three FGSM perturbation strengths:

```text
ε = 0.01
ε = 0.05
ε = 0.10
````

---

## Folder Contents

* `FGSM.ipynb`
* `FGSM_RECALL_TRADEOFF.ipynb`

---

## Requirements

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib pillow scikit-learn
```

---

## Dataset

The notebooks use the CARLA test dataset:

```text
test/
│
├── labels.csv
└── rgb-front/
```

Download the dataset from:

[https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing](https://drive.google.com/drive/folders/1xrseTTZAbfLkhn83mjTpT9XJNUzhd1m0?usp=sharing)

Update the dataset path in the notebooks before running the experiments.

---

## Pre-trained Models

The experiments use the trained models from the previous weeks:

```text
pedestrian_model.pth
Vehicle_model.pth
Traffic_Light_model.pth
```

Download the trained models from:

[https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing](https://drive.google.com/drive/folders/1_3ErNfpXeV3PUcuBYP56xpVimwahzHDi?usp=sharing)

Update the model paths in the notebooks before running them.

---

## How to Reproduce the Results

### 1. FGSM Visualization

Run:

```text
FGSM.ipynb
```

This notebook demonstrates the FGSM attack on a correctly classified pedestrian image.

The notebook:

* Loads the trained pedestrian model.
* Selects a correctly classified test image.
* Calculates the loss with respect to the input image.
* Computes the gradient of the loss.
* Applies the FGSM perturbation.
* Generates adversarial images using different perturbation strengths.
* Displays the clean image alongside adversarial versions.

The perturbation strengths are:

```text
ε = 0.01
ε = 0.05
ε = 0.10
```

---

### 2. FGSM Attack

The FGSM attack is implemented as:

```python
def fgsm_attack(image, epsilon, data_grad):
    sign_grad = data_grad.sign()

    perturbed_image = (
        image + epsilon * sign_grad
    )

    perturbed_image = torch.clamp(
        perturbed_image,
        0,
        1
    )

    return perturbed_image
```

The attack modifies the input in the direction of the sign of the gradient.

The perturbed image is clipped to the valid image range `[0, 1]`.

---

### 3. FGSM Recall Trade-off

Run:

```text
FGSM_RECALL_TRADEOFF.ipynb
```

This notebook evaluates the effect of FGSM perturbations on recall for all three perception models.

The evaluation function:

```text
evaluate_fgsm_recall()
```

performs the following steps:

1. Loads the selected trained model.
2. Loads the CARLA test dataset.
3. Randomly selects up to 300 test images.
4. Calculates clean recall.
5. Generates FGSM adversarial examples.
6. Evaluates recall for each perturbation strength.
7. Stores the clean and adversarial recall values.

---

## Models Evaluated

### Pedestrian Model

The pedestrian model is evaluated using:

```text
Model:
pedestrian_model.pth

Label:
has_pedestrian
```

Run:

```python
pedestrian_results = evaluate_fgsm_recall(
    MODEL_PATH=".../pedestrian_model.pth",
    DATASET_PATH=".../test",
    LABEL_COLUMN="has_pedestrian"
)
```

---

### Vehicle Model

The vehicle model is evaluated using:

```text
Model:
Vehicle_model.pth

Label:
has_vehicle
```

Run:

```python
VEHICLE_results = evaluate_fgsm_recall(
    MODEL_PATH=".../Vehicle_model.pth",
    DATASET_PATH=".../test",
    LABEL_COLUMN="has_vehicle"
)
```

---

### Traffic-Light Model

The traffic-light model is evaluated using:

```text
Model:
Traffic_Light_model.pth

Label:
has_traffic_light
```

Run:

```python
TL_results = evaluate_fgsm_recall(
    MODEL_PATH=".../Traffic_Light_model.pth",
    DATASET_PATH=".../test",
    LABEL_COLUMN="has_traffic_light"
)
```

---

## FGSM Perturbation Levels

The recall experiment evaluates:

```python
epsilons = [0.01, 0.05, 0.1]
```

The interpretation is:

| Epsilon | Perturbation Strength |
| ------: | --------------------- |
|    0.01 | Small                 |
|    0.05 | Moderate              |
|    0.10 | Stronger              |

---

## Recall Evaluation

For every model, the notebook first calculates recall on the clean test subset.

It then generates adversarial examples for each epsilon value and calculates recall again.

The recall is calculated using:

```python
recall_score(
    y_true,
    y_pred,
    zero_division=0
)
```

The resulting dictionary has the following structure:

```text
{
    "Clean": ...,
    0.01: ...,
    0.05: ...,
    0.1: ...
}
```

This makes it possible to compare clean recall with recall under increasing adversarial perturbation.

---

## Sampling

The recall trade-off experiment uses:

```text
Maximum sample size = 300 images
```

The images are selected randomly from the test dataset:

```python
indices = random.sample(
    range(len(dataset)),
    min(sample_size, len(dataset))
)
```

The evaluation therefore operates on a randomly selected subset of up to 300 test images.

---

## Expected Outputs

Running `FGSM.ipynb` produces:

* One clean image.
* One adversarial image for ε = 0.01.
* One adversarial image for ε = 0.05.
* One adversarial image for ε = 0.10.
* A visual comparison of the clean and perturbed inputs.

Running `FGSM_RECALL_TRADEOFF.ipynb` produces:

* Clean recall for the pedestrian model.
* FGSM recall for ε = 0.01, 0.05 and 0.10.
* Clean recall for the vehicle model.
* FGSM recall for ε = 0.01, 0.05 and 0.10.
* Clean recall for the traffic-light model.
* FGSM recall for ε = 0.01, 0.05 and 0.10.

---

## Notes

* The experiments use the previously trained ResNet18 perception models.
* No model retraining is performed.
* FGSM perturbations are generated using the gradient of the binary cross-entropy-with-logits loss with respect to the input image.
* The perturbed images are clipped to the range `[0, 1]`.
* The prediction threshold is **0.5** after applying the sigmoid function.
* The recall experiment uses up to **300 randomly selected test images**.
* The same test dataset is used for the pedestrian, vehicle and traffic-light models.
* The experiment evaluates recall rather than accuracy because missed positive detections are particularly important for safety-critical perception tasks.
* The three perturbation strengths are **ε = 0.01, 0.05 and 0.10**.

---



```
```
