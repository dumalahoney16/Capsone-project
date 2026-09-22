# Viva Presentation: Explainable AI-Based Diabetic Retinopathy Screening

**Format**: 15 Slides with Key Bullet Points & Speaker Notes  
**Designed for**: B.Tech Final Year Project Examination / Viva Voce  

---

## Slide 1: Title Slide
- **Project Title**: Explainable AI-Based Diabetic Retinopathy Screening Using Deep Learning
- **Subtitle**: A Student-Friendly Decision-Support Screening System for Rural & Underserved Settings
- **Domain**: Artificial Intelligence / Medical Image Processing / Deep Learning / Explainable AI (XAI)
- **Candidate Name(s)**: [Student Name(s)]
- **Guide / Supervisor**: [Supervisor Name]
- **Department**: Computer Science and Engineering / Artificial Intelligence
- **Institution**: [College / University Name]

*Speaker Notes: "Respected examiners, good morning. Today I present our final-year B.Tech project on Explainable AI-Based Diabetic Retinopathy Screening Using Deep Learning."*

---

## Slide 2: Introduction
- **What is Diabetic Retinopathy (DR)?**
  - A progressive microvascular complication of diabetes mellitus affecting retinal blood vessels.
  - Causes microaneurysms, hemorrhages, hard exudates, and abnormal vessel growth.
- **The Global & Indian Crisis**:
  - Over 77 million diabetic adults in India ("Diabetes Capital of the World").
  - Leading cause of preventable visual impairment and adult blindness.
- **Why Early Screening Matters**:
  - Early-stage DR is completely asymptomatic.
  - Early detection and glycemic management reduce severe blindness risk by over 90%.

*Speaker Notes: "DR begins silently with microvascular changes in the retina. If caught early, 90% of blindness is preventable; however, patients only visit clinics when symptoms appear, which is often too late."*

---

## Slide 3: Problem Statement
- **Severe Shortage of Eye Specialists in Rural India**:
  - 70% of India's population lives in rural areas, yet >75% of ophthalmologists practice in Tier-1 cities.
  - Doctor-to-patient ratio in rural districts exceeds 1:250,000.
- **The "Black-Box" Dilemma in Healthcare AI**:
  - Standard deep learning algorithms output diagnoses without explanation.
  - Doctors cannot verify whether predictions are based on genuine disease markers or image noise/artifacts.
- **Resource Constraints**:
  - High-end GPU hardware and specialized software are absent in rural Primary Health Centres (PHCs).

*Speaker Notes: "Our problem statement targets two hurdles: the extreme shortage of rural eye care specialists, and the black-box nature of conventional AI that creates clinical mistrust."*

---

## Slide 4: Existing Systems vs Limitations
- **Manual Ophthalmologist Examination**:
  - Gold standard, but slow, expensive, and inaccessible in remote regions.
- **Classical Handcrafted Computer Vision**:
  - Matched filtering, morphological transforms, edge detection.
  - Sensitive to lighting changes, camera artifacts, and pigment variations; high false positives.
- **Traditional Deep CNNs (ResNet, VGG)**:
  - High diagnostic accuracy, but act as opaque black boxes with zero interpretability.
  - Heavy computational footprint, unsuitable for edge/web deployment.

*Speaker Notes: "Existing computer vision either suffers from rigid manual feature design or functions as an opaque, heavy neural network that offers no visual justification for its predictions."*

---

## Slide 5: Proposed System Overview
- **RetinaVision XAI Architecture**:
  - Web-based, lightweight medical screening triage platform.
  - Uses **EfficientNetB0** transfer learning (ImageNet pretrained weights).
  - Integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** for transparent visual explanations.
  - Accessible **Streamlit** user interface accessible over low-bandwidth web connections.
  - Provides neutral, medically sound screening triage and referral guidance.

*Speaker Notes: "Our proposed solution unites transfer learning for accurate 5-stage classification with Grad-CAM for visual transparency, deployed through a lightweight Streamlit interface."*

---

## Slide 6: Project Objectives
1. Build a multi-class deep learning model for the 5 clinical stages of Diabetic Retinopathy.
2. Implement data preprocessing and augmentation for retinal fundus images.
3. Integrate Grad-CAM to visually highlight anatomical decision regions.
4. Construct an intuitive web application with multiclass probability graphs and confidence metrics.
5. Provide clear ethical boundaries: screening decision-support, not medical diagnosis.

*Speaker Notes: "Our key objectives are accurate 5-stage classification, visual interpretability via Grad-CAM, a simple web UI, and strict adherence to ethical medical screening standards."*

---

## Slide 7: Technology Stack
- **Programming**: Python 3.10
- **Deep Learning Framework**: TensorFlow 2.x & Keras
- **Backbone Architecture**: EfficientNetB0 (Compound scaling)
- **Explainable AI (XAI)**: Grad-CAM (using `tf.GradientTape`)
- **Computer Vision**: OpenCV (headless) & Pillow
- **Data & Evaluation**: Scikit-Learn, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Web Application Deployment**: Streamlit

*Speaker Notes: "We selected a modern, robust Python stack featuring TensorFlow for deep learning, OpenCV for image operations, and Streamlit for rapid web deployment."*

---

## Slide 8: Dataset & Clinical Grading
- **Primary Dataset**: APTOS 2019 Blindness Detection (Kaggle).
- **International Clinical Diabetic Retinopathy Disease Severity Scale**:
  - **Class 0 (No DR)**: Clean retina, healthy macula and optic disc.
  - **Class 1 (Mild DR)**: Microaneurysms only.
  - **Class 2 (Moderate DR)**: More than microaneurysms, less than severe (exudates, hemorrhages).
  - **Class 3 (Severe DR)**: 4-2-1 rule (>20 intraretinal hemorrhages in each quadrant).
  - **Class 4 (Proliferative DR)**: Neovascularization, vitreous hemorrhage.
- **Reproducibility Utility**: Includes `create_sample_data.py` for synthetic testing.

*Speaker Notes: "The dataset follows the international clinical 5-stage scale, ranging from healthy retina to proliferative retinopathy with neovascularization."*

---

## Slide 9: Methodology & Pipeline Flow
1. **Acquisition & Preprocessing**:
   - Fundus image upload -> RGB validation -> Bilinear resize to 224×224 -> Float32 batch tensor.
2. **Data Augmentation**:
   - Random horizontal/vertical flip, random rotation (8%), random zoom (8%).
3. **Two-Phase Transfer Learning**:
   - *Phase 1*: Base model frozen; classification head trained with Adam ($lr=10^{-3}$).
   - *Phase 2*: Top 20 layers unfrozen for fine-tuning with reduced learning rate ($lr=10^{-4}$).
4. **Grad-CAM Saliency Computation**:
   - Backward gradient flow to the final `top_conv` convolutional layer.
5. **Screening Decision Delivery**:
   - Class, confidence, probability distribution, Grad-CAM overlay, and referral advice.

*Speaker Notes: "Our workflow covers standardized preprocessing, two-stage fine-tuning with callbacks, Grad-CAM extraction, and clinical referral generation."*

---

## Slide 10: Deep Learning Model Architecture
- **Backbone**: EfficientNetB0 (Pretrained on ImageNet, 224×224×3 input).
- **Why EfficientNetB0?**
  - Uses compound coefficient scaling (depth, width, resolution).
  - High accuracy with only ~4.0M parameters (compared to ResNet-50's 25M+).
- **Classification Head**:
  - Global Average Pooling 2D
  - Batch Normalization
  - Dropout (0.30)
  - Dense (128 units, ReLU activation)
  - Dropout (0.20)
  - Dense Output (5 units, Softmax activation)
- **Model Storage**: Saved as `models/dr_model.keras`.

*Speaker Notes: "EfficientNetB0 offers the optimal balance between diagnostic feature extraction and parameter efficiency, making it ideal for edge and web execution."*

---

## Slide 11: Explainable AI — Grad-CAM
- **Why Grad-CAM?**
  - Translates abstract neural activations into human-interpretable thermal heatmaps.
- **How it Works Mathematically**:
  1. Computes gradients of target class output $y^c$ with respect to feature maps $A^k$.
  2. Pools gradients to calculate importance weights $\alpha_k^c$.
  3. Computes weighted linear combination: $\sum \alpha_k^c A^k$.
  4. Applies ReLU to isolate features with positive influence.
  5. Blends heatmap over original fundus image using OpenCV `COLORMAP_JET` ($\alpha = 0.45$).
- **Medical Communication**:
  - Highlights *salient predictive regions*; does not claim definitive proof of lesions.

*Speaker Notes: "Grad-CAM traces gradients backwards to the final convolutional feature maps, creating a visual heat map where red and orange indicate regions driving the model's prediction."*

---

## Slide 12: Streamlit Web Application
- **Beginner-Friendly, Intuitive Layout**:
  - Retinal fundus image uploader (supports JPG, JPEG, PNG).
  - Built-in sample selector for 1-click viva testing.
  - "Analyze Retinal Image" action trigger.
- **Output Components**:
  - Color-coded severity badge (Green -> Red).
  - Confidence percentage score.
  - Multiclass probability bar chart.
  - Side-by-side comparison: Original Fundus vs. Grad-CAM Overlay.
  - Actionable clinical referral advisory (routine checkup vs. specialist referral).
  - Prominent ethical medical disclaimer banner.

*Speaker Notes: "Our Streamlit application provides a clean, responsive interface designed for community health workers, featuring side-by-side explainability and clear referral guidance."*

---

## Slide 13: Results & Clinical Evaluation
- **Evaluation Metrics**:
  - Overall Accuracy: Benchmark classification rate.
  - Precision: Minimizes false alarms and unnecessary tertiary hospital referrals.
  - **Recall (Sensitivity)**: Paramount in medical screening—missing a patient with severe DR can cause irreversible blindness.
  - F1-Score: Harmonic balance on imbalanced datasets.
- **Generated Artifacts**:
  - `outputs/confusion_matrix.png` (evaluates inter-stage confusion).
  - `outputs/training_accuracy.png` & `outputs/training_loss.png`.
  - `outputs/gradcam.jpg`.

*Speaker Notes: "In medical screening, high Recall is essential because a false negative carries catastrophic clinical consequences for the patient."*

---

## Slide 14: Limitations & Future Scope
- **Current Limitations**:
  - Resolution downsampling (224×224) may compress microscopic microaneurysms (<25 microns).
  - Model relies on good image illumination and focus.
  - Coarse $7\times 7$ Grad-CAM resolution rather than pixel-level lesion boundaries.
  - Not yet validated in prospective rural clinical trials.
- **Future Scope**:
  - Multi-task screening for Glaucoma and Macular Degeneration.
  - Vision Transformer (ViT) patch networks for microscopic lesion detection.
  - TensorFlow Lite (TFLite) mobile quantization for offline smartphone fundus cameras.
  - Integration with rural Ayushman Bharat telemedicine registries.

*Speaker Notes: "Future improvements include multi-disease classification, Vision Transformers for higher resolution, and TFLite deployment on handheld smartphone cameras."*

---

## Slide 15: Conclusion
- **Key Takeaways**:
  - Developed a complete, functional Explainable AI screening tool for Diabetic Retinopathy.
  - EfficientNetB0 provides fast, accurate multi-class severity grading.
  - Grad-CAM bridges the clinical trust gap by making model reasoning transparent.
  - Streamlit interface empowers non-physician health workers in underserved settings.
  - Strictly adheres to medical AI ethics: screening decision-support, not clinical diagnosis.
- **Thank You!** Questions and viva discussion are welcome.

*Speaker Notes: "In conclusion, our project demonstrates that combining efficient deep learning with Explainable AI can bring accessible, trustworthy retinal screening to underserved populations. Thank you."*
