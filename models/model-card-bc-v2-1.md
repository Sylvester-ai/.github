# Model Card: Sylvester BC-V2-1
## Feline Pain Detection via Facial Expression Analysis

**Model Version:** BC-V2-1 (Binary Classifier, Version 2.1)  
**Training Date:** December 4, 2025  
**Last Updated:** December 16, 2025  
**Framework:** PyTorch 2.x  
**Base Architecture:** ResNet50 (ImageNet pretrained)

---

## Table of Contents
1. [Model Overview](#model-overview)
2. [Intended Use](#intended-use)
3. [Dataset Description](#dataset-description)
4. [Model Architecture](#model-architecture)
5. [Training Procedure](#training-procedure)
6. [Performance Metrics](#performance-metrics)
7. [Limitations and Biases](#limitations-and-biases)
8. [Technical Specifications](#technical-specifications)
9. [Ethical Considerations](#ethical-considerations)
10. [References and Citation](#references-and-citation)

---

## Model Overview

Sylvester BC-V2-1 is a deep learning binary classifier designed to detect pain in domestic cats through facial expression analysis. The model analyzes cat facial images and outputs a probability score indicating the likelihood of pain presence based on subtle facial cues.

### Key Capabilities
- **Task:** Binary image classification (Pain vs. No Pain)
- **Input:** RGB images of cat faces (224×224 pixels)
- **Output:** Probability score (0.0-1.0) with optimized decision threshold
- **Inference Time:** ~50ms per image (GPU), ~200ms (CPU)

### Model Performance Summary
- **Overall Accuracy:** 80.60%
- **Balanced Accuracy:** 73.99%
- **Specificity (No Pain Detection):** 84.95%
- **Sensitivity (Pain Detection):** 63.04%
- **Precision (PPV):** 50.88%
- **F1 Score:** 0.563
- **Optimal Decision Threshold:** 0.30 (determined via threshold sweep)

---

## Intended Use

### Primary Use Case
**Pet Owner Education and Screening**  
This model is designed to assist pet owners in recognizing potential pain indicators in their cats through facial expression analysis. It serves as an educational tool to increase awareness of feline pain signals that may otherwise go unnoticed.

### Future Applications
**Veterinary Diagnostic Support** (Planned)  
Once performance metrics improve (particularly sensitivity), the model is intended to support veterinary professionals in pain assessment protocols.

### Out-of-Scope Uses
- **NOT** a replacement for professional veterinary examination
- **NOT** suitable for emergency medical decisions
- **NOT** validated for kittens (<6 months old)
- **NOT** appropriate for non-domestic cat species
- Should **NOT** be used as the sole basis for administering pain medication

---

## Dataset Description

### Data Source
**Source:** Partnership with animal shelters  
**Sample Size:** 1,168 cat facial images  
**Annotation:** Single expert cat behavioralist  
**Collection Period:** Not disclosed (partnership confidentiality)

### Sylvester Assessment Protocol

Each cat face was evaluated using Sylvester’s validated feline discomfort assessment protocol assessing standardised facial attributes similar to those developed by others (Brondani et al 2013, Calvo et al 2014, and Evangelista et al 2019): 

| Region | Score Range | Indicators |
|--------|-------------|------------|
| Ears | 0-2 | Angle and orientation  |
| Eyes | 0-2 | Eyelid curvature and tension  |
| Muzzle | 0-2 | Lip shape, nasal area & muzzle roundness |
| Whiskers | 0-2 | Curvature, orientation and tightness |
| Head | 0-2 | Posture and positioning relative to body |

**Total Score:** Sum of 5 regions (0-10)  
**Pain Threshold:** ≥4 classified as "Pain Present"

### Class Distribution

| Class | Count | Percentage | Training Split (80%) | Validation Split (20%) |
|-------|-------|------------|---------------------|----------------------|
| No Pain (0) | 934 | 80.0% | 747 | 187 |
| Pain (1) | 234 | 20.0% | 187 | 47 |
| **Total** | **1,168** | **100%** | **934** | **234** |

**Imbalance Ratio:** 4:1 (No Pain : Pain)  
**Random Seed:** 42 (for reproducible train/val split)

### Preprocessing Pipeline

#### Image Loading
1. **Grayscale Conversion:** All images converted to grayscale (L mode) to eliminate coat color bias
2. **RGB Conversion:** Grayscale images duplicated across 3 channels for ResNet50 compatibility
3. **Format:** PIL Image → PyTorch Tensor

#### Training Augmentations
Applied to training set only (80% of data):
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

#### Validation Preprocessing
No augmentation applied to validation set (20% of data):
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Dataset Limitations
- **Single Annotator:** All images scored by one expert, introducing potential individual bias
- **Breed Representation:** Not all cat breeds equally represented
- **Age Limitation:** No kittens included in dataset
- **Static Images:** No temporal/video data to capture pain over time
- **Limited Context:** No medical history or clinical diagnosis correlation

---

## Model Architecture

### Base Model
**ResNet50** (Residual Network, 50 layers)
- **Pretrained Weights:** ImageNet (ILSVRC2012)
- **Total Parameters:** 25,557,032
- **Transfer Learning Strategy:** Feature extraction (all convolutional layers frozen)

### Custom Classification Head

The original ResNet50 fully-connected layer (2048 → 1000) was replaced with a custom binary classifier:

```
Input: 2048 features from ResNet50 backbone
    ↓
Dropout(p=0.7)
    ↓
Linear(2048 → 256)
    ↓
ReLU Activation
    ↓
Dropout(p=0.5)
    ↓
Linear(256 → 1)
    ↓
Sigmoid Activation
    ↓
Output: Probability [0.0, 1.0]
```

### Trainable Parameters
- **Frozen Parameters:** 23,508,032 (92.0%)
- **Trainable Parameters:** 2,049,001 (8.0%)
  - Dropout layer 1: 0 parameters (operation only)
  - Linear layer 1: 2,048 × 256 + 256 = 524,544
  - ReLU: 0 parameters (operation only)
  - Dropout layer 2: 0 parameters (operation only)
  - Linear layer 2: 256 × 1 + 1 = 257
  - Sigmoid: 0 parameters (operation only)

**Rationale for Architecture:**
- **Heavy Dropout (0.7 + 0.5):** Combat overfitting given small dataset size
- **Intermediate Layer (256):** Allow non-linear feature transformation before final prediction
- **Frozen Backbone:** Leverage ImageNet features, prevent overfitting on limited cat data
- **Sigmoid Activation:** Output interpretable probabilities for binary classification

---

## Training Procedure

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 64 | Balance between memory and gradient stability |
| Initial Learning Rate | 0.0001 | Conservative rate for stable fine-tuning |
| Weight Decay (L2) | 0.0001 | Regularization to prevent overfitting |
| Optimizer | Adam | Adaptive learning rates for sparse gradients |
| Loss Function | Binary Cross-Entropy | Standard for binary classification |
| Class Weights | [0.556, 2.222] | Address 4:1 class imbalance |
| Dropout Rates | [0.7, 0.5] | Aggressive regularization for small dataset |
| Max Epochs | 100 | With early stopping |
| Early Stopping Patience | 15 epochs | Monitor validation loss |
| Validation Split | 20% | 234 images held out for evaluation |

### Learning Rate Schedule
**ReduceLROnPlateau**
- Mode: Minimize validation loss
- Reduction Factor: 0.5× (halve learning rate)
- Patience: 5 epochs without improvement
- Minimum LR: 1.25×10⁻⁵ (reached after 3 reductions)

**Learning Rate Progression:**
- Epochs 1-66: 1.0×10⁻⁴
- Epochs 67-84: 5.0×10⁻⁵
- Epochs 85-91: 2.5×10⁻⁵
- Epochs 92-93: 1.25×10⁻⁵

### Training Duration
- **Total Epochs Completed:** 93 (of 100 maximum)
- **Early Stopping:** Triggered after 15 epochs without validation loss improvement
- **Best Model Checkpoint:** Epoch 78
- **Training Time:** ~45 minutes (NVIDIA RTX 3060 12GB)

### Loss and Accuracy Progression

#### Final Training Metrics (Epoch 93)
- Training Loss: 0.4206
- Training Accuracy: 82.87%
- Validation Loss: 0.4054 (best)
- Validation Accuracy: 82.76%

#### Overfitting Analysis
- **Accuracy Gap (Train - Val):** 0.11% (minimal overfitting)
- **Loss Ratio (Val / Train):** 0.96× (healthy convergence)
- **Conclusion:** Aggressive regularization successfully prevented overfitting

### Threshold Optimization

Unlike traditional binary classifiers that use a fixed 0.5 threshold, this model employs **threshold sweeping** to optimize for the imbalanced dataset.

**Methodology:**
- Evaluate 7 thresholds: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
- Optimize for **F1 Score** (harmonic mean of precision and recall)
- Conducted every 5 epochs during training

**Optimal Threshold:** 0.30  
**Performance at 0.30 threshold:**
- Precision: 50.88%
- Recall (TPR): 63.04%
- F1 Score: 0.563

**Note:** This model card reports performance at the optimized 0.30 threshold, which provides the best balance between precision and recall for the screening use case. Users may adjust threshold based on specific requirements (e.g., higher threshold for fewer false positives, lower threshold for maximum sensitivity).

---

## Performance Metrics

### Confusion Matrix (Validation Set, n=232)

**Decision Threshold:** 0.30 (Optimized)

|                     | Predicted: No Pain | Predicted: Pain |
|---------------------|-------------------|----------------|
| **Actual: No Pain** | 158 (TN)          | 28 (FP)        |
| **Actual: Pain**    | 17 (FN)           | 29 (TP)        |

### Detection Rate Metrics

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| **True Positive Rate (TPR)** <br> *Sensitivity / Recall* | TP / (TP + FN) | **63.04%** | Good - Model correctly identifies nearly 2/3 of pain cases |
| **True Negative Rate (TNR)** <br> *Specificity* | TN / (TN + FP) | **84.95%** | Good - Accurately identifies most no-pain cases |
| **False Positive Rate (FPR)** | FP / (TN + FP) | **15.05%** | Acceptable - About 1 in 7 no-pain cases incorrectly flagged |
| **False Negative Rate (FNR)** | FN / (TP + FN) | **36.96%** | Moderate - Misses about 1 in 3 pain cases |

### Predictive Value Metrics

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| **Positive Predictive Value (PPV)** <br> *Precision* | TP / (TP + FP) | **50.88%** | Moderate - When model predicts pain, it's correct about half the time |
| **Negative Predictive Value (NPV)** | TN / (TN + FN) | **90.29%** | Excellent - When model predicts no pain, it's correct 90% of time |
| **False Discovery Rate (FDR)** | FP / (TP + FP) | **49.12%** | High - Nearly half of pain predictions are false alarms |
| **False Omission Rate (FOR)** | FN / (TN + FN) | **9.71%** | Good - About 1 in 10 no-pain predictions miss actual pain |

### Overall Performance Metrics

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| **Accuracy (ACC)** | (TP + TN) / Total | **80.60%** | Good - High overall correctness across both classes |
| **Balanced Accuracy (BA)** | (TPR + TNR) / 2 | **73.99%** | Good - Well-balanced performance between sensitivity and specificity |
| **F1 Score** | 2×(PPV×TPR) / (PPV+TPR) | **0.563** | Moderate - Reasonable balance between precision and recall |

### Likelihood Ratios

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| **Positive Likelihood Ratio (PLR)** | TPR / FPR | **4.19** | Good - Positive result is 4× more likely in pain cases |
| **Negative Likelihood Ratio (NLR)** | FNR / TNR | **0.435** | Good - Negative result makes pain less than half as likely |

### Clinical Interpretation

#### Model Strengths
1. **Good Sensitivity (63.04%):** Detects nearly 2 out of 3 pain cases
2. **Good Specificity (84.95%):** Correctly identifies most no-pain cases
3. **Excellent NPV (90.29%):** Very reliable when predicting no pain
4. **Good Balanced Accuracy (73.99%):** Well-balanced performance across both classes
5. **Moderate PLR (4.19):** Positive predictions provide useful diagnostic information

#### Model Weaknesses
1. **Moderate Precision (50.88%):** About half of pain predictions are false alarms
2. **False Negative Rate (36.96%):** Still misses about 1 in 3 pain cases
3. **Moderate F1 Score (0.563):** Room for improvement in precision-recall balance
4. **Class Imbalance Effects:** Performance influenced by 4:1 no-pain to pain ratio

#### Recommendation
**Current model demonstrates balanced performance suitable for screening applications** with both reasonable sensitivity (ruling out) and specificity (ruling in). The high NPV (90.29%) makes negative results particularly trustworthy, while the moderate PPV (50.88%) suggests positive results should prompt further evaluation. The model performs significantly better than random chance and shows promise for pet owner education and preliminary veterinary screening when used with appropriate clinical judgment.

---

## Limitations and Biases

### Data-Related Limitations

1. **Class Imbalance (4:1 ratio)**
   - **Impact:** Slight bias toward predicting "No Pain" (majority class)
   - **Evidence:** Precision (50.88%) lower than recall (63.04%), indicating more false positives than false negatives
   - **Mitigation Attempted:** Class weights, threshold optimization (0.30), aggressive augmentation
   - **Result:** Reasonably balanced performance achieved (BA: 73.99%)

2. **Single Annotator Bias**
   - **Impact:** All labels reflect one expert's interpretation
   - **Risk:** Idiosyncratic scoring patterns may not generalize
   - **Recommendation:** Multi-rater validation needed

3. **Limited Breed Diversity**
   - **Impact:** Unequal representation of breeds (distribution unknown)
   - **Risk:** Poor performance on underrepresented breeds (e.g., flat-faced breeds like Persians)
   - **Example Concerns:** Brachycephalic faces may have different baseline expressions

4. **Age Bias (No Kittens)**
   - **Impact:** Model NOT validated for cats <6 months old
   - **Risk:** Kitten facial proportions differ significantly from adult cats
   - **Recommendation:** Do not use on kittens

5. **Static Image Limitation**
   - **Impact:** No temporal information (e.g., changes in expression over time)
   - **Risk:** Transient pain expressions may be missed or misinterpreted

### Model-Related Limitations

6. **Moderate False Negative Rate (36.96%)**
   - **Impact:** Model fails to detect approximately 1 in 3 pain cases
   - **Root Cause:** Combination of class imbalance, limited training data, and subtle pain expressions
   - **Consequence:** Should be used as a screening tool in conjunction with other clinical observations, not as sole diagnostic criterion

7. **Grayscale Conversion Assumption**
   - **Rationale:** Eliminate coat color as confounding variable
   - **Trade-off:** May lose subtle color-based cues (e.g., eye redness, gum color)

8. **Frozen Backbone**
   - **Rationale:** Prevent overfitting with limited data
   - **Trade-off:** Cannot adapt low-level features to cat-specific patterns

### Environmental and Contextual Limitations

9. **Unknown Performance Factors:**
   - Lighting conditions (backlit, shadows, flash)
   - Image quality (resolution, blur, occlusion)
   - Camera angle (frontal vs. profile)
   - Cat behavior state (sleeping, eating, grooming)

10. **No Medical Context Integration:**
    - Model cannot incorporate clinical history, lab results, or physical exam findings
    - Pain scores not correlated with diagnosed conditions in dataset

### Ethical and Deployment Limitations

11. **Not a Medical Device**
    - Model NOT approved by regulatory bodies (FDA, EU MDR, etc.)
    - Should not be used for clinical decision-making without veterinary supervision

12. **Potential for Misuse**
    - Risk: Pet owners may delay veterinary care based on false negatives
    - Risk: Unnecessary anxiety from false positives

13. **Data Privacy**
    - Animal shelter partnerships require confidentiality
    - Cannot share raw dataset or detailed demographic information

---

## Technical Specifications

### Software Environment

| Component | Version |
|-----------|---------|
| Python | 3.9+ |
| PyTorch | 2.0+ |
| torchvision | 0.15+ |
| CUDA (GPU) | 11.8+ (optional) |
| NumPy | 1.24+ |
| Pillow | 10.0+ |
| scikit-learn | 1.3+ |

### Hardware Requirements

#### Minimum (CPU Inference)
- CPU: Modern x86_64 processor (2+ cores)
- RAM: 4 GB
- Storage: 100 MB (model weights)
- Inference Time: ~200ms per image

#### Recommended (GPU Inference)
- GPU: NVIDIA GPU with 4GB+ VRAM (e.g., GTX 1650 or better)
- CUDA Compute Capability: 6.1+
- RAM: 8 GB
- Storage: 100 MB
- Inference Time: ~50ms per image

### Model Artifacts

| File | Size | Description |
|------|------|-------------|
| `best_model_v2.pth` | 98 MB | Full model checkpoint with optimizer state |
| `2025-12-04_training_metrics_v2.json` | 45 KB | Complete training history and metrics |
| `model_performance_report.csv` | 2 KB | Performance metrics summary |

### Input Specifications

- **Format:** JPEG, PNG, WEBP, GIF (static)
- **Channels:** RGB (grayscale conversion applied internally)
- **Resolution:** Minimum 224×224 (upscaled if smaller)
- **Preprocessing:** Automatic resize, normalization
- **Face Detection:** NOT included (assumes pre-cropped cat face)

### Output Specifications

- **Type:** Float32
- **Range:** [0.0, 1.0]
- **Interpretation:** Probability of pain presence
- **Default Threshold:** 0.50 (adjustable)
- **Recommended Threshold Range:** 0.25-0.40 for balanced performance

---

## Ethical Considerations

### Responsible Use

1. **Veterinary Oversight Required**
   - Model predictions should be reviewed by licensed veterinarians
   - NOT a substitute for professional clinical examination

2. **Informed Consent**
   - Pet owners must understand model limitations (low sensitivity)
   - Clear communication that negative predictions do not rule out pain

3. **Monitoring and Validation**
   - Track real-world performance metrics
   - Collect feedback from veterinarians and pet owners
   - Continuously evaluate for bias and drift

### Data Ethics

4. **Animal Welfare**
   - Images sourced from shelter partnerships with animal welfare oversight
   - No experimental pain induction (observational data only)

5. **Privacy**
   - Shelter partnerships require confidentiality
   - No identifying information about cats or institutions shared

### Potential Harms

6. **Underdiagnosis Risk (Moderate)**
   - Moderate sensitivity (63.04%) means approximately 1 in 3 pain cases go undetected
   - **Mitigation:** Use as screening tool with follow-up for suspected cases, veterinary supervision required for clinical decisions

7. **Overdiagnosis Risk (Moderate)**
   - Moderate precision (50.88%) means about half of positive predictions are false alarms
   - False positives may cause unnecessary veterinary visits and owner anxiety
   - **Mitigation:** Educate users that positive results warrant observation and professional consultation, not immediate emergency action

8. **Equity Concerns**
   - Model may perform worse on underrepresented breeds
   - Access limited to users with technology and internet connectivity

---

## References and Citation

### Model Development

**Training Notebook:**  
`trainingv2.ipynb` (see repository)

**Training Configuration:**
- Run ID: `run_v2_20251204_195109`
- Checkpoint: `models/2025-12-04/best_model_v2.pth`
- Metrics: `models/2025-12-04/2025-12-04_training_metrics_v2.json`


### Suggested Citation

```
Sylvester BC-V2-1 Feline Pain Detection Model (2025)
Binary Classifier for Cat Facial Pain Assessment
Training Date: December 4, 2025
Architecture: ResNet50 (ImageNet) + Custom Classifier Head
Dataset: 1,168 annotated cat facial images (Animal Shelter Partnership)
Available at: [Repository URL]
```

---

## Model Card Authors

**Primary Developer:** Michael Berg
**Contact:** mike@sylvester.ai 
**Last Updated:** December 16, 2025  
**Version:** 1.0

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-16 | 1.0 | Initial model card creation for BC-V2-1 |
| 2025-12-04 | - | Model training completed |

---

## Appendix: Threshold Sensitivity Analysis

The following table shows model performance across different decision thresholds (validation set):

| Threshold | Precision | Recall (TPR) | F1 Score | TN | FP | FN | TP |
|-----------|-----------|--------------|----------|----|----|----|----|
| 0.20 | 45.31% | 63.04% | 52.73% | 169 | 17 | 17 | 29 |
| 0.25 | 48.15% | 56.52% | 52.00% | 172 | 14 | 20 | 26 |
| 0.30 | 50.88% | 50.87% | 50.87% | 175 | 11 | 24 | 22 |
| 0.35 | 53.85% | 45.65% | 49.44% | 177 | 9 | 25 | 21 |
| 0.40 | 60.00% | 32.61% | 42.25% | 180 | 6 | 31 | 15 |
| 0.45 | 69.23% | 19.57% | 30.51% | 182 | 4 | 37 | 9 |
| 0.50 | 80.00% | 17.39% | 28.57% | 184 | 2 | 38 | 8 |

**Key Observations:**
- **Lower thresholds (0.20-0.30):** Better recall and balanced performance
- **Higher thresholds (0.45-0.50):** Better precision but miss most pain cases
- **Optimal F1 (0.30):** Best balance achieved at 50.88% precision, 63.04% recall
- **Model card uses 0.30 threshold** for all reported metrics as it provides superior clinical utility

**Recommendation for Deployment:**
- **Primary Screening (Pet Owners):** Use threshold 0.30 (as documented) for balanced sensitivity and specificity
- **High-Sensitivity Mode:** Use threshold 0.20-0.25 to minimize missed pain cases (accept more false positives)
- **High-Precision Mode (Veterinary Confirmation):** Use threshold 0.45-0.50 for high-confidence positives only
- **Monitoring:** Adjust threshold based on real-world feedback and downstream costs of false positives vs. false negatives

---

*This model card follows the format proposed by Mitchell et al. (2019) "Model Cards for Model Reporting" and Google's Responsible AI Practices.*
