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
  
---

# Week 6 – LLM Safety, Calibration and Backdoor Attacks

- Explored LLM safety concepts such as temperature sampling, benchmark evaluation, LLM-as-Judge, and prompt injection attacks.
- Studied AI agent safety, trajectory evaluation, and adversarial behaviour in autonomous systems.
- Implemented temperature scaling on the pedestrian detector using different temperature values (T = 0.5, 1.0, 2.0) to analyze confidence calibration and safety-trigger behaviour.
- Observed that temperature scaling changes model confidence while accuracy remained mostly unchanged.
- Implemented a backdoor attack by poisoning 10% of pedestrian images using a 10×10 red trigger square and flipped labels.
- Retrained the pedestrian detector on the poisoned dataset and evaluated Clean Recall and Attack Success Rate (ASR).
- Observed that the model maintained similar clean performance while achieving a high ASR (~76%), demonstrating hidden malicious behaviour in ML systems.

# Week 7 – Explainability and Model Diagnostics

- Learned the importance of Explainable AI (XAI) in ML Safety.
- Understood Local and Global Explainability.
- Studied Occlusion, Saliency Maps, CAM, and Grad-CAM.
- Learned Chain-of-Thought Fidelity and Simulatability concepts.
- Applied the Occlusion method to CARLA classification models.
- Generated heatmaps for correct and incorrect predictions.
- Used explainability to analyze model behavior and failures.
- Tested models under nighttime conditions (distribution shift).
- Observed that correct predictions mostly focused on the target objects.
- Observed that misclassified samples often focused on background regions.
- Found that nighttime images caused more attention to lighting and road reflections.
- Noticed a decrease in both model performance and explanation quality under nighttime conditions.
- Learned how explainability helps identify spurious features and generalization issues.

# Week 8: Anomaly Detection and OOD Detection
- Learned Anomaly Detection, In-Distribution (ID), and Out-of-Distribution (OOD) concepts.
- Studied the risks of silent failures in safety-critical ML systems.
- Learned MSP (Maximum Softmax Probability) as a baseline OOD detector.
- Implemented and evaluated MSP-based OOD detection on CARLA datasets.
- Explored feature-based OOD detection using k-NN.
- Compared OOD detection performance using AUROC.
- Understood the difference between OOD issues and generalization issues.
- Extended the STPA safety analysis to include OOD-related hazards and safety constraints.
- Vehicle and traffic-light models became less confident on fog and night images.
- The pedestrian model remained highly confident on OOD inputs.
- MSP achieved an AUROC of 0.8168.
- k-NN achieved an AUROC of 0.7799.
- Night images were easier to detect as OOD than fog images.
- Different-town images were treated as a generalization problem rather than OOD.
# WEEK 9
What I Learned:
- Learned the concept of adversarial examples and how small perturbations can fool ML models.
- Understood the difference between adversarial examples and OOD inputs.
- Studied FGSM and PGD attacks and how they optimize the input instead of model weights.
- Learned the concepts of targeted vs. untargeted attacks and perturbation budget (ε).
- Explored adversarial training and its trade-offs.
- Extended the STPA safety analysis to include adversarial threats.

Observations:
- FGSM attacks significantly affected all three CARLA models.
- The traffic-light model was the most vulnerable to adversarial perturbations.
- The vehicle model achieved high recall at larger ε values by predicting the positive class for all inputs, showing that recall alone can be misleading.
- The pedestrian model showed unstable behaviour under stronger attacks.
- Adversarial robustness improves safety but does not eliminate all risks.


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
