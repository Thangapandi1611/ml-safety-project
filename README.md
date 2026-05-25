# Introduction to Machine Learning Safety

This repository contains the weekly exercises, implementations, and learning progress completed as part of the *Introduction to Machine Learning Safety* course using the CARLA autonomous driving dataset.

---

# Week 1 – Introduction to ML Safety & Security

In the first week, the focus was understanding:
- What Machine Learning Safety means
- Difference between ML Safety and ML Security
- Why high accuracy alone is not enough in safety-critical systems
- Challenges in autonomous driving systems
- Importance of reliability, robustness, and validation in ML models

Key concepts learned:
- Safety vs Security
- Traditional Software vs ML Systems
- Risk and failure in AI systems
- Real-world consequences of model errors

---

# Week 3 – Hazard Analysis and STPA

In the Third week, the focus moved towards safety engineering concepts and applying them to the CARLA autonomous driving system.

Concepts explored:
- Hazard
- Loss
- Risk
- Unsafe Control Actions (UCA)
- Safety Constraints
- STPA (System-Theoretic Process Analysis)

Activities completed:
- Identified hazards in autonomous driving scenarios
- Created losses and risk analysis
- Defined Unsafe Control Actions
- Developed safety constraints for the perception system
- Applied STPA concepts to the CARLA dataset

This week helped connect machine learning with real-world safety engineering practices.

---

# Week 4 – Building Perception Models

In the Fourth week, three separate binary classification models were developed using PyTorch and ResNet18.

Models built:
1. Pedestrian Detection Model
2. Vehicle Detection Model
3. Traffic Light Detection Model

Tasks completed:
- Dataset preprocessing and loading
- Transfer learning using ResNet18
- Model training and validation
- Training and validation loss analysis
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

The pedestrian model analysis particularly highlighted why recall is more important than accuracy in autonomous driving safety applications.

---
# Week 5 – Testing

In the Fifth week, we I learnt ML testing and other evaluation ways , K projection and Distribution shift.
Tasks  completed:
-Evaluated the models on different test data which includes Foggy, night, town images
- Designed a Test Suite from Safety Constraints in Ex:2
- Evaluated K-projection with ODD


# Technologies Used

- Python
- PyTorch
- ResNet18
- Google Colab
- Scikit-learn
- Matplotlib
- CARLA Dataset

---

# Key Learning Outcome

This course helped understand that building an ML model is only one part of the process.  
The more important challenge is ensuring that the model behaves safely and reliably under different real-world operating conditions.
