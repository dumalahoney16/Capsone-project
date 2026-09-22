# Explainable AI-Based Diabetic Retinopathy Screening Using Deep Learning

A student-friendly, research-grade medical image screening application designed to assist in early detection and triage of **Diabetic Retinopathy (DR)** from retinal fundus photographs, with a specific focus on low-resource and rural healthcare settings in India.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![EfficientNet](https://img.shields.io/badge/Architecture-EfficientNetB0-brightgreen.svg)
![XAI](https://img.shields.io/badge/Explainability-Grad--CAM-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Table of Contents
1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [System Architecture](#system-architecture)
7. [Methodology](#methodology)
8. [Model Architecture & Training](#model-architecture--training)
9. [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
10. [Streamlit Application](#streamlit-application)
11. [Results and Evaluation](#results-and-evaluation)
12. [Environment Setup & Installation](#environment-setup--installation)
13. [How to Run (Step-by-Step)](#how-to-run-step-by-step)
14. [Testing Checklist](#testing-checklist)
15. [Limitations](#limitations)
16. [Future Scope](#future-scope)
17. [Disclaimer](#disclaimer)

---

## Abstract

Diabetic Retinopathy (DR) is a leading cause of preventable adult blindness globally and in India, where over 77 million individuals live with diabetes. Early detection through regular screening is vital, yet rural regions face severe shortages of ophthalmologists and diagnostic infrastructure. This project presents an end-to-end, Explainable AI-based screening decision-support system. Using transfer learning with **EfficientNetB0**, the system classifies digital fundus images into five clinical severity stages: No DR, Mild, Moderate, Severe, and Proliferative DR. To avoid the opacity of deep neural networks in healthcare, **Gradient-weighted Class Activation Mapping (Grad-CAM)** is implemented to visually highlight the anatomical regions contributing to each classification. Deployed via an intuitive **Streamlit** web application, the system offers rapid inference, confidence scores, probability distributions, visual explanations, and neutral clinical referral guidance for rural community screening.

---

## Problem Statement

1. **Silent Progression**: In its initial stages, Diabetic Retinopathy is largely asymptomatic; by the time patients notice visual impairment, irreversible retinal damage has often occurred.
2. **Ophthalmologist Shortage in Rural India**: Over 70% of India's population resides in rural areas, whereas more than 75% of eye-care specialists practice in major urban cities.
3. **Black-Box AI Trust Gap**: Conventional Convolutional Neural Networks (CNNs) output diagnostic predictions without clinical justification, creating mistrust among healthcare workers and clinicians.

---

## Objectives

1. Develop a lightweight deep-learning pipeline for multi-class DR grading based on **EfficientNetB0**.
2. Preprocess and augment retinal fundus images to ensure robustness against camera variations.
3. Implement **Grad-CAM** to render transparent, heatmapped visual explanations for model predictions.
4. Build a beginner-friendly, accessible **Streamlit** web user interface with multiclass probabilities, metric cards, and referral guidance.
5. Provide comprehensive academic deliverables (project report, presentation slides, viva Q&A, and testing guide).

---

## Technologies Used

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming and pipeline integration |
| **Deep Learning** | TensorFlow / Keras | EfficientNetB0 transfer learning and inference |
| **Explainable AI** | Grad-CAM (Gradient Tape) | Gradient-based visual saliency heatmaps |
| **Computer Vision** | OpenCV & Pillow | Fundus preprocessing, resizing, and alpha overlay |
| **Data Processing** | NumPy & Pandas | Numerical tensors and probability dataframes |
| **Metrics & Eval** | Scikit-learn | Precision, Recall, F1-Score, Confusion Matrix |
| **Visualization** | Matplotlib & Seaborn | Loss/accuracy curves and confusion matrix heatmaps |
| **Web Application** | Streamlit | Interactive browser-based screening interface |

---

## Dataset

This project utilizes the **APTOS 2019 Blindness Detection** dataset (Asia Pacific Tele-Ophthalmology Society) accessible via Kaggle.

### Five Clinical Classes:
* **Class 0 → No Diabetic Retinopathy** (`dataset/No_DR`)
* **Class 1 → Mild Diabetic Retinopathy** (`dataset/Mild`)
* **Class 2 → Moderate Diabetic Retinopathy** (`dataset/Moderate`)
* **Class 3 → Severe Diabetic Retinopathy** (`dataset/Severe`)
* **Class 4 → Proliferative Diabetic Retinopathy** (`dataset/Proliferate_DR`)

### How to Download and Setup the Real Kaggle Dataset:
1. Go to Kaggle: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) or the resized 224×224 version.
2. Download `train.csv` and the training images zip.
3. Extract the archive into your working environment.
4. Organize images into the five class folders matching `diagnosis` (0 to 4):
   - `0` → `dataset/No_DR/`
   - `1` → `dataset/Mild/`
   - `2` → `dataset/Moderate/`
   - `3` → `dataset/Severe/`
   - `4` → `dataset/Proliferate_DR/`
5. Verify that each directory contains valid `.png` or `.jpg` images.

### Quick Student Testing (Synthetic Data Generator):
If internet bandwidth is limited or you want to verify the pipeline immediately without downloading 8.8 GB:
```powershell
python src/create_sample_data.py
```
This generates representative synthetic fundus images across all five classes with simulated optic discs, retinal vessels, microaneurysms, and exudates.

---

## System Architecture

```text
                  +-----------------------------------+
                  |      User / Healthcare Worker     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |     Streamlit Web Interface       |
                  |     (app.py - Port 8501)          |
                  +-----------------+-----------------+
                                    |
                        [Upload Retinal Fundus]
                                    |
                                    v
                  +-----------------------------------+
                  |    Image Preprocessing Pipeline   |
                  |  - RGB Convert, 224x224 Resize    |
                  |  - Float32 Batch Tensor Formation |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |   EfficientNetB0 Deep Learning    |
                  |  - Pretrained ImageNet Weights    |
                  |  - Global Avg Pool + Dropout      |
                  |  - Dense 128 + Softmax Head       |
                  +-----------------+-----------------+
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
          +-------------------------+  +--------------------------+
          | 5-Class DR Prediction   |  | Explainable AI: Grad-CAM |
          | - Predicted Class       |  | - Target Feature Layer   |
          | - Confidence Score (%)  |  | - Backprop Gradients     |
          | - Probability Breakdown |  | - Saliency Heatmap       |
          +------------+------------+  +------------+-------------+
                       |                            |
                       +-------------+--------------+
                                     |
                                     v
                  +-----------------------------------+
                  |     Screening Decision Support    |
                  |  - Side-by-Side Saliency Overlay  |
                  |  - Multiclass Probability Chart   |
                  |  - Clinical Referral Advisory     |
                  |  - Medical Disclaimer Notice      |
                  +-----------------------------------+
```

---

## Methodology

1. **Acquisition & Standardisation**: Input fundus images of variable dimensions are cast to RGB, scaled to 224×224 pixels, and conditioned for neural network ingestion.
2. **Transfer Learning Feature Extraction**: Pretrained EfficientNetB0 captures multi-scale retinal textures (fovea, vascular geometry, microaneurysms, and cotton-wool spots).
3. **Staged Fine-Tuning**:
   - *Phase 1*: Base model frozen; classification head optimized with Adam (`lr = 0.001`).
   - *Phase 2*: Top 20 layers unfrozen with reduced learning rate (`lr = 0.0001`) for retinal domain adaptation.
4. **Regularization & Callbacks**:
   - `EarlyStopping` prevents overfitting.
   - `ReduceLROnPlateau` lowers learning rate when validation loss plateaus.
   - `ModelCheckpoint` preserves the top-performing checkpoint at `models/dr_model.keras`.
5. **Explainability Extraction**: Grad-CAM captures gradients flowing from the predicted severity score into the final convolutional feature maps (`top_conv`), creating an attention heatmap blended at $\alpha = 0.45$.
6. **Clinical Screening Triaging**: Predictions are translated into actionable, neutral triage advisories (routine annual checkup vs. specialist referral).

---

## Model Architecture & Training

```text
Input (224 × 224 × 3)
   │
   ├─► Data Augmentation (RandomFlip, RandomRotation 8%, RandomZoom 8%)
   │
   ├─► EfficientNetB0 Backbone (weights='imagenet', include_top=False)
   │
   ├─► GlobalAveragePooling2D
   │
   ├─► BatchNormalization
   │
   ├─► Dropout (0.30)
   │
   ├─► Dense (128 units, ReLU activation)
   │
   ├─► Dropout (0.20)
   │
   └─► Dense (5 units, Softmax activation) -> [P0, P1, P2, P3, P4]
```

---

## Explainable AI (Grad-CAM)

Deep learning models are notoriously prone to "shortcut learning" (e.g., focusing on camera artifacts rather than ocular pathology). Grad-CAM computes:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

Where:
- $y^c$ is the score for Diabetic Retinopathy class $c$.
- $A^k$ is the feature activation map of channel $k$ in the final convolutional layer.
- $\alpha_k^c$ captures the importance weight of channel $k$ for class $c$.
- $\text{ReLU}$ ensures that only features positively contributing to the target class are displayed.

The resulting heatmap is scaled, colored using OpenCV's `COLORMAP_JET`, and blended with the original retinal photograph.

---

## Streamlit Application

The interactive web interface is designed with simple language suited for community health volunteers and medical students:
- **Interactive File Upload**: Accepts `.jpg`, `.jpeg`, and `.png` fundus photos.
- **Demo Selection**: Built-in sample selector allows immediate 1-click evaluation.
- **Metric Badges**: Color-coded severity badge (Green, Cyan, Amber, Orange, Crimson).
- **Probability Breakdown**: Horizontal distribution chart across all 5 clinical stages.
- **Dual Visual Inspection**: Side-by-side display of original fundus photograph and Grad-CAM saliency overlay.
- **Referral Advisory**: Clear, responsible guidance emphasizing specialist consultation.
- **Architecture & Performance Tabs**: Direct inspection of training curves and confusion matrix.

---

## Results and Evaluation

The model is evaluated using Scikit-Learn:
* **Accuracy**: Overall classification correctness.
* **Precision**: Measures true positive rate out of all positive predictions, minimizing unnecessary rural-to-urban hospital referrals.
* **Recall (Sensitivity)**: Critical for medical screening. A false negative (missing a patient with severe DR) can lead to irreversible blindness.
* **F1-Score**: Harmonic mean of Precision and Recall, accounting for class imbalance.
* **Confusion Matrix**: Visual breakdown of stage-wise misclassifications, saved to `outputs/confusion_matrix.png`.

---

## Environment Setup & Installation

### 1. Prerequisites
- Windows 10/11 (or macOS/Linux)
- Python 3.10 or compatible version installed (ensure **"Add Python to PATH"** was checked during installation).
- Visual Studio Code installed.

### 2. Project Directory Setup in VS Code
Open PowerShell or Command Prompt in your desired folder and clone/navigate to the project:
```powershell
cd C:\Users\Honey\OneDrive\Desktop\AICW\PROJECT
```

### 3. Create Virtual Environment
```powershell
python -m venv venv
```

### 4. Activate Virtual Environment
- **Windows PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an Execution Policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*
- **Windows Command Prompt (cmd)**:
  ```cmd
  venv\Scripts\activate.bat
  ```

### 5. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Common TensorFlow Installation Troubleshooting:
* **TensorFlow DLL / CUDA issues on Windows**:
  TensorFlow 2.12+ uses native CPU instructions automatically. If GPU issues occur, CPU mode runs seamlessly for EfficientNetB0 inference and training without requiring CUDA toolkit configuration.
* **Long Path Limitations on Windows**:
  Enable long file paths in Windows registry or install Python in a root directory like `C:\Python310`.
* **NumPy 2.x Compatibility**:
  TensorFlow 2.15/2.16 requires NumPy `< 2.0.0`. The provided `requirements.txt` locks this specification (`numpy>=1.24.0,<2.0.0`).

---

## How to Run (Step-by-Step)

Follow this sequence to run and verify the entire project:

### Step 1: Generate or Place Dataset
If you have Kaggle APTOS data, place images in `dataset/No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`.
Otherwise, generate sample data immediately:
```powershell
python src/create_sample_data.py
```

### Step 2: Train the EfficientNetB0 Model
```powershell
python src/train.py
```
*Loads datasets, displays class imbalance, trains the head, fine-tunes top layers, saves `models/dr_model.keras`, and outputs accuracy/loss curves to `outputs/`.*

### Step 3: Evaluate Model Performance
```powershell
python src/evaluate.py
```
*Generates precision/recall metrics and saves the confusion matrix to `outputs/confusion_matrix.png`.*

### Step 4: Test Grad-CAM Standalone
```powershell
python src/gradcam.py
```
*Produces an explainable visual overlay and saves it to `outputs/gradcam.jpg`.*

### Step 5: Test Single Image Prediction
```powershell
python src/predict.py
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Testing Checklist

| # | Test Scenario | Input / Action | Expected Result |
| :-: | :--- | :--- | :--- |
| 1 | Normal Retinal Fundus | Upload healthy fundus image | Class 0 (No DR), high confidence, green badge |
| 2 | Advanced DR Image | Upload proliferative fundus | Class 3 or 4, red badge, specialist referral advisory |
| 3 | Grad-CAM Overlay | Click "Analyze Retinal Image" | Side-by-side original and thermal attention map |
| 4 | Multiclass Chart | Complete analysis | 5 horizontal bars summing to ~100% |
| 5 | Missing Image | Click analyze without image | Gentle instructional notification; no crash |
| 6 | Non-Image File Upload | Upload text or PDF file | Streamlit file rejection filter; clean error message |
| 7 | Missing Model File | Delete/rename `dr_model.keras` | Warning banner with guidance to run `train.py` |
| 8 | Sample Selector | Pick image from dropdown | Immediate image preview and analysis capability |
| 9 | Large/Small Image Size | Upload 512x512 or 1024x1024 | Bilinear auto-resizing to 224x224 without distortion |
| 10 | Browser Responsiveness | Resize window or mobile view | Clean responsive layout with sidebar and tabs |

---

## Limitations

1. **Resolution Downscaling**: Clinical fundus cameras capture images at 2000×2000+ pixels. Downsampling to 224×224 pixels can occasionally obscure minute microaneurysms (< 25 microns).
2. **Quality Dependence**: Poor illumination, corneal cataracts, or improper pupillary dilation can reduce prediction confidence.
3. **Non-Clinical Certification**: The model is trained on public research datasets and has not completed multi-center clinical trials in rural Indian primary healthcare centers.
4. **Grad-CAM Resolution**: The final feature map of EfficientNetB0 is $7 \times 7$, providing coarse anatomical localization rather than pixel-level lesion segmentation.

---

## Future Scope

1. **Multi-Task Learning**: Concurrently predicting Diabetic Retinopathy, Glaucoma, and Age-Related Macular Degeneration (AMD) from a single fundus capture.
2. **High-Resolution Patch Networks**: Implementing Vision Transformers (ViT) or patch-based attention to detect microscopic lesions without global downsampling.
3. **Edge Deployment**: Quantizing model weights using TensorFlow Lite (TFLite) for deployment on battery-powered mobile fundus cameras and offline Android devices.
4. **External Clinical Validation**: Prospective validation studies in partnership with rural community health centers across Indian states.

---

## Disclaimer

> **IMPORTANT MEDICAL NOTICE**:
> This software is created strictly for **educational, academic research, and screening decision-support purposes**. It does **NOT** provide a definitive medical diagnosis and should never replace professional clinical examination, optical coherence tomography (OCT), or evaluation by a qualified ophthalmologist. Neither the authors nor the academic institution assume liability for any medical decisions made based on this application.
