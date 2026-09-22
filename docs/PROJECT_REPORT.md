# Final-Year B.Tech Project Report

# Explainable AI-Based Diabetic Retinopathy Screening Using Deep Learning

**Degree**: Bachelor of Technology (B.Tech) in Computer Science and Engineering / Artificial Intelligence  
**Academic Year**: 2025–2026  
**Focus Area**: Medical Image Analysis, Transfer Learning, Explainable AI (XAI), Rural Health Screening  

---

## Table of Contents
1. [Chapter 1: Introduction](#chapter-1-introduction)
2. [Chapter 2: Literature Survey](#chapter-2-literature-survey)
3. [Chapter 3: Problem Statement](#chapter-3-problem-statement)
4. [Chapter 4: Objectives](#chapter-4-objectives)
5. [Chapter 5: Proposed System](#chapter-5-proposed-system)
6. [Chapter 6: Methodology](#chapter-6-methodology)
7. [Chapter 7: System Architecture](#chapter-7-system-architecture)
8. [Chapter 8: Implementation](#chapter-8-implementation)
9. [Chapter 9: Results and Evaluation](#chapter-9-results-and-evaluation)
10. [Chapter 10: Limitations](#chapter-10-limitations)
11. [Chapter 11: Future Scope](#chapter-11-future-scope)
12. [Chapter 12: Conclusion](#chapter-12-conclusion)

---

## Chapter 1: Introduction

### 1.1 Medical Background
Diabetes Mellitus has emerged as one of the most critical public health challenges of the 21st century. According to the International Diabetes Federation (IDF), India is home to over 77 million individuals with diabetes, earning it the moniker of the "diabetes capital of the world." One of the most prevalent and sight-threatening microvascular complications of prolonged hyperglycemia is **Diabetic Retinopathy (DR)**.

Diabetic Retinopathy damages the delicate microvascular network of the retina—the neurosensory tissue lining the inner posterior surface of the eye responsible for converting light into neural signals. Persistent high blood glucose induces endothelial cell loss, capillary basement membrane thickening, and pericyte degeneration. This leads to capillary occlusions, microaneurysms (small balloon-like outpouchings), fluid leakage (hard exudates), retinal hemorrhages, and retinal ischemia. In advanced stages, retinal hypoxia triggers the release of vascular endothelial growth factor (VEGF), stimulating fragile, abnormal new blood vessels (neovascularization) that frequently bleed into the vitreous humor and induce tractional retinal detachment, culminating in irreversible blindness.

### 1.2 The Clinical Screening Imperative
Crucially, Diabetic Retinopathy remains largely asymptomatic during its early stages (Mild and Moderate Non-Proliferative DR). Patients typically retain 20/20 visual acuity even while significant retinal microvascular alterations are underway. By the time visual symptoms manifest (such as blurring, floaters, or dark scotomas), the disease has often progressed to proliferative DR or diabetic macular edema (DME), where therapeutic interventions (such as pan-retinal photocoagulation or anti-VEGF intravitreal injections) can preserve remaining sight but cannot restore lost neural tissue.

Global clinical guidelines uniformly advocate annual dilated retinal fundus examinations for all diabetic individuals. When detected early, prompt glycemic control and timely ophthalmological referral reduce the risk of severe visual impairment by more than 90%.

### 1.3 The Rural Healthcare Landscape in India
While regular screening is standard in advanced healthcare settings, achieving universal screening in India is constrained by a profound structural disparity:
- **Demographic Distribution**: Approximately 65–70% of India's population lives in rural and semi-urban communities.
- **Specialist Concentration**: Over 75% of ophthalmologists and nearly 90% of specialized vitreoretinal surgeons practice in Tier-1 and Tier-2 metropolitan cities.
- **Doctor-to-Patient Ratio**: In rural Indian districts, the ratio of eye specialists to patients frequently exceeds 1 : 250,000, making routine in-person specialist screening impossible for rural agricultural workers and marginalized populations.

Consequently, automated, portable, and explainable AI-based screening solutions deployed at Primary Health Centres (PHCs) or Ayushman Bharat Health and Wellness Centres (HWCs) present an immense socio-technical opportunity. Such systems allow non-physician frontline health workers to capture fundus photographs, obtain instantaneous risk stratification, and route high-risk individuals to tertiary ophthalmology centers.

---

## Chapter 2: Literature Survey

### 2.1 Evolution of Diabetic Retinopathy Analysis
Computer-assisted screening for Diabetic Retinopathy has evolved across three technological epochs:
1. **Classical Handcrafted Feature Extraction (1995–2012)**:
   Early systems relied on mathematical morphology, Gabor filters, matched filters, and Hough transforms to segment blood vessels, followed by thresholding to isolate bright lesions (hard exudates) and dark lesions (hemorrhages). Feature vectors were classified using Support Vector Machines (SVM) or Random Forests. These methods suffered from high false-positive rates due to illumination non-uniformities, physiological pigmentation variations, and artifacts from uncooperative patients.
2. **Deep Convolutional Neural Networks (2012–2019)**:
   The advent of AlexNet, VGGNet, and ResNet revolutionized retinal analysis. Seminal work by Gulshan et al. (JAMA 2016) demonstrated that a deep CNN trained on over 128,000 fundus photographs could achieve sensitivity and specificity comparable to board-certified ophthalmologists for binary referable DR detection.
3. **Compound Scaling & Modern Efficient Backbones (2019–Present)**:
   While heavy networks like ResNet-152 and DenseNet-201 achieved strong benchmark scores, their computational burden (millions of parameters and billions of FLOPs) made them ill-suited for rural edge devices or low-bandwidth web portals. Tan & Le (2019) introduced **EfficientNet**, utilizing compound scaling to simultaneously scale network depth, width, and resolution with high FLOP efficiency.

### 2.2 The Explainability Void in Medical Deep Learning
Despite stellar statistical accuracy, real-world deployment of deep learning in clinical medicine has met resistance due to the "black-box" dilemma. Traditional CNNs output a single softmax probability without revealing which anatomical features drove the prediction. In high-stakes healthcare:
- Clinicians cannot verify whether the network focused on genuine pathology (e.g., hard exudates or microaneurysms) or spurious confounders (e.g., camera lens dust, boundary vignetting, or patient eyelashes).
- Medical professionals bear legal and ethical accountability for diagnostic outcomes and will not act on recommendations they cannot inspect.

### 2.3 Explainable AI (XAI) and Grad-CAM
To restore clinical transparency, Selvaraju et al. (2017) formulated **Gradient-weighted Class Activation Mapping (Grad-CAM)**. Unlike earlier CAM formulations that necessitated modifying the network architecture with a Global Average Pooling layer directly preceding the softmax, Grad-CAM utilizes the gradient of the target class score flowing into the final convolutional feature layer of any CNN backbone. By retaining spatial awareness and combining it with directional class-specific gradients, Grad-CAM produces coarse visual saliency maps highlighting the exact pixel clusters that governed the network’s final verdict.

---

## Chapter 3: Problem Statement

1. **High Preventable Blindness in Rural India**: Millions of diabetic individuals in rural and semi-urban Indian regions lack access to ophthalmologists, leading to late-stage detection of Diabetic Retinopathy and preventable vision loss.
2. **Diagnostic Opacity**: Existing automated screening algorithms function as opaque black boxes, preventing community health workers and reviewing clinicians from validating the reasoning behind automated classifications.
3. **Resource Constraints**: High-capacity deep neural architectures require high-end GPU workstations unavailable in rural community health settings. A lightweight, efficient, and accessible web-based screening workflow is urgently needed.

---

## Chapter 4: Objectives

1. **Design and Implement a Lightweight Transfer Learning Pipeline**:
   Adapt the **EfficientNetB0** architecture pretrained on ImageNet for 5-stage Diabetic Retinopathy severity grading (No DR, Mild, Moderate, Severe, and Proliferative DR).
2. **Implement an Explainable AI Module**:
   Integrate **Grad-CAM** using TensorFlow GradientTape to produce visual heatmaps overlaid on the input fundus image, pinpointing predictive regions.
3. **Develop a Student-Friendly, Responsive Web Interface**:
   Build an accessible **Streamlit** user interface featuring image upload, real-time inference, confidence scoring, multiclass probability distributions, and side-by-side saliency inspection.
4. **Establish Ethical and Responsible Clinical Triage Rules**:
   Ensure the application provides neutral, medically responsible referral guidance rather than definitive diagnostic or treatment directives, adhering to medical AI ethics.
5. **Formulate a Reproducible Academic Benchmark**:
   Provide complete runnable code, synthetic test generator, evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix), and academic documentation.

---

## Chapter 5: Proposed System

The proposed system, **RetinaVision XAI**, is an assistive, explainable tele-screening decision-support pipeline designed for low-resource clinics.

```text
+--------------------------------------------------------------------------+
|                            PROPOSED SYSTEM FLOW                          |
+--------------------------------------------------------------------------+
|  1. Input Fundus Image (Standard digital fundus camera capture)          |
|  2. Preprocessing (RGB conversion, resizing to 224x224, tensor batching) |
|  3. Data Augmentation (Rotation, flips, zoom for invariance)             |
|  4. EfficientNetB0 Feature Extraction (ImageNet pretrained weights)      |
|  5. Classification Head (Global Avg Pool, Dropout, Dense, Softmax)       |
|  6. 5-Class Severity Stratification (Stage 0 to Stage 4)                 |
|  7. Grad-CAM Computation (Backpropagation to final convolutional layer)  |
|  8. Saliency Heatmap Rendering (OpenCV COLORMAP_JET, alpha blending)     |
|  9. Streamlit Dashboard Delivery (Metrics, Probabilities, Referral advice)|
+--------------------------------------------------------------------------+
```

### Key Advantages of the Proposed System:
- **Computational Efficiency**: EfficientNetB0 requires only ~4.0 million parameters, allowing CPU-based inference within 1–2 seconds.
- **Interpretability by Design**: Clinicians and screening operators inspect Grad-CAM heatmaps alongside predictions.
- **Accessibility**: Zero local installation required for end-users when hosted on web or local Intranet.

---

## Chapter 6: Methodology

### 6.1 Clinical Staging Standard
The system adheres to the **International Clinical Diabetic Retinopathy Disease Severity Scale**:
- **Class 0 (No DR)**: Absence of retinal lesions or vascular abnormalities.
- **Class 1 (Mild Non-Proliferative DR)**: Presence of microaneurysms only.
- **Class 2 (Moderate Non-Proliferative DR)**: More than just microaneurysms but less than severe NPDR (cotton-wool spots, venous beading, hard exudates).
- **Class 3 (Severe Non-Proliferative DR)**: Characterized by the 4-2-1 rule (intraretinal hemorrhages in all 4 quadrants, venous beading in 2 quadrants, or intraretinal microvascular abnormalities [IRMA] in 1 quadrant).
- **Class 4 (Proliferative DR)**: Marked by neovascularization on the disc or elsewhere, or vitreous/preretinal hemorrhage.

### 6.2 Data Acquisition & Preprocessing
Input images are sourced from the APTOS 2019 Blindness Detection dataset (or generated via `src/create_sample_data.py` for student verification). Images undergo:
1. Format standardization to 3-channel RGB.
2. Spatial downsampling to $224 \times 224$ pixels using bilinear interpolation.
3. Batch tensor conversion into `float32`.

### 6.3 Data Augmentation
To simulate imaging inconsistencies in rural clinics (e.g., patient movement, imperfect pupil dilation, varying illumination), real-time on-the-fly augmentation is integrated:
- Random horizontal and vertical flips.
- Small rotations ($\pm 8\%$).
- Random zooms ($\pm 8\%$).

### 6.4 Training Strategy
Training is executed in two staged phases:
- **Phase 1 (Transfer Learning / Frozen Base)**: The EfficientNetB0 backbone is frozen. Only the classification head (Dense layers and Batch Normalization) is trained using Adam optimizer ($lr = 10^{-3}$) with Sparse Categorical Cross-Entropy.
- **Phase 2 (Fine-Tuning)**: The top 20 convolutional layers of EfficientNetB0 are unfrozen and trained at an attenuated learning rate ($lr = 10^{-4}$) to adapt high-level feature detectors specifically to retinal patterns.

---

## Chapter 7: System Architecture

### 7.1 Detailed Component Breakdown
1. **User Interface Layer (`app.py`)**:
   Built with Streamlit, handling file upload, user interaction, interactive charts, and disclaimer rendering.
2. **Preprocessing & Inference Engine (`src/predict.py`)**:
   Loads the cached model from `models/dr_model.keras`, transforms raw input arrays into tensor representations, and extracts Softmax class probabilities.
3. **Explainable AI Module (`src/gradcam.py`)**:
   Employs TensorFlow `GradientTape` to compute class-specific activation gradients with respect to the `top_conv` layer of EfficientNetB0.
4. **Evaluation & Reporting Module (`src/evaluate.py`)**:
   Computes multiclass precision, recall, F1-scores, and generates the confusion matrix heatmap saved in `outputs/confusion_matrix.png`.

---

## Chapter 8: Implementation

### 8.1 Software & Hardware Specifications
- **Operating System**: Windows 11 (64-bit)
- **Language**: Python 3.10
- **Frameworks**: TensorFlow 2.x, Keras, Streamlit, OpenCV, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- **Code Organization**: Modular structure separating training (`train.py`), evaluation (`evaluate.py`), prediction (`predict.py`), and visualization (`gradcam.py`).

### 8.2 Grad-CAM Mathematical Formulation
Given an input image $I$, the model predicts a logits score $y^c$ for class $c$. The importance weight $\alpha_k^c$ for feature map $k$ of the final convolutional layer $A$ is:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i,j}^k}$$

where $Z = u \times v$ represents the spatial area of the feature map. The final 2D saliency heatmap $L^c$ is computed as:

$$L^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

Applying the Rectified Linear Unit (ReLU) ensures that the map only reflects features whose presence increases the score of class $c$, ignoring features that suppress it.

---

## Chapter 9: Results and Evaluation

### 9.1 Evaluation Metrics
Medical screening models cannot be judged on raw classification accuracy alone due to prevalent class imbalance. We assess:
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Measures the fraction of actual DR cases detected. In medical screening, maximizing recall for Stage 2, 3, and 4 is critical to ensure no sight-threatening cases are overlooked.
- **Precision (Positive Predictive Value)**: $\frac{TP}{TP + FP}$ — Measures the reliability of positive classifications, guarding against overwhelming tertiary eye centers with false referrals.
- **F1-Score**: Harmonic mean of Precision and Recall: $\frac{2 \cdot P \cdot R}{P + R}$.

### 9.2 Confusion Matrix Analysis
The confusion matrix (`outputs/confusion_matrix.png`) maps ground-truth severity grades against predicted grades. A strong diagonal indicates accurate classification, while off-diagonal elements illustrate clinical overlap primarily between adjacent grades (e.g., Mild vs. Moderate NPDR), which is consistent with inter-observer grading variability among human ophthalmologists.

---

## Chapter 10: Limitations

1. **Resolution Constraints**: Downsampling high-resolution fundus images ($2000 \times 2000$) to $224 \times 224$ pixels reduces computational demands but risks obliterating isolated microaneurysms (< 25 $\mu$m).
2. **Dependence on Image Quality**: Poor focus, corneal opacity, or media haze can reduce model confidence.
3. **Coarse Saliency Maps**: Because the final feature map of EfficientNetB0 is $7 \times 7$, Grad-CAM provides regional attention heatmaps rather than fine, pixel-level lesion segmentations.
4. **Lack of Clinical Multi-Center Validation**: The system is trained on public benchmark repositories and has not undergone prospective multi-center clinical validation across varied rural Indian demographics.

---

## Chapter 11: Future Scope

1. **Multi-Disease Screening**: Extend the architecture to perform multi-label classification covering Glaucoma (cup-to-disc ratio analysis) and Age-Related Macular Degeneration (AMD).
2. **Patch-Based Attention Networks**: Incorporate Vision Transformers (ViT) or high-resolution multi-crop patch networks to preserve microscopic lesion fidelity.
3. **Embedded Edge AI**: Quantize the model to TensorFlow Lite (TFLite) or ONNX format for deployment on handheld, smartphone-coupled fundus cameras running offline without cloud connectivity.
4. **Integration with Tele-Ophthalmology Networks**: Interface the application with rural Ayushman Bharat HWC electronic health records for automated patient registry and referral dispatch.

---

## Chapter 12: Conclusion

This project successfully demonstrates the design, implementation, and evaluation of an **Explainable AI-based Diabetic Retinopathy Screening System** tailored for low-resource and rural Indian healthcare environments. By leveraging **EfficientNetB0** transfer learning, the system achieves rapid, computationally lightweight classification across all five DR stages. The integration of **Grad-CAM** addresses the fundamental "black-box" barrier in healthcare by providing transparent visual explanations for every screening decision. Deployed through an intuitive **Streamlit** dashboard, the application bridges the gap between complex deep learning models and frontline healthcare workers, offering a practical framework for early visual impairment prevention.
