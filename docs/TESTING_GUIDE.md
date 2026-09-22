# Testing Guide & Quality Assurance Checklist
## Explainable AI-Based Diabetic Retinopathy Screening

This document outlines the **10-Point Comprehensive Testing Matrix** for validating the data pipeline, deep learning model, Grad-CAM explainability module, and Streamlit user interface.

---

## 1. Quick Verification Command Sequence

Before testing individual test cases, ensure your environment is initialized and dependencies are installed:

```powershell
# 1. Navigate to project root
cd C:\Users\Honey\OneDrive\Desktop\AICW\PROJECT

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Step 1: Generate or verify sample dataset
python src/create_sample_data.py

# 4. Step 2: Train model (or verify existing models/dr_model.keras)
python src/train.py

# 5. Step 3: Run evaluation & confusion matrix
python src/evaluate.py

# 6. Step 4: Verify Grad-CAM standalone
python src/gradcam.py

# 7. Step 5: Test CLI prediction
python src/predict.py

# 8. Step 6: Launch Streamlit Web Application
streamlit run app.py
```

---

## 2. Comprehensive 10-Point QA Test Matrix

### Test 1: Valid Normal Retinal Fundus (Class 0)
* **Input**: Fundus image of a healthy retina (`dataset/No_DR/sample_no_dr_001.png`).
* **Procedure**: Upload via the Streamlit interface or select from the sample dropdown. Click **"Analyze Retinal Image"**.
* **Expected Output**:
  - Predicted Class: **No Diabetic Retinopathy** (Class 0).
  - Badge Color: **Green**.
  - Confidence Score: High ($> 60\%$).
  - Referral Recommendation: Routine annual checkup advisory.
* **Pass/Fail Criteria**: Pass if Class 0 is predicted and no referral alert is triggered.

---

### Test 2: Severe / Proliferative Diabetic Retinopathy (Class 3 or 4)
* **Input**: Fundus image displaying extensive hemorrhages or neovascularization (`dataset/Proliferate_DR/sample_proliferate_dr_001.png`).
* **Procedure**: Upload image and click **"Analyze Retinal Image"**.
* **Expected Output**:
  - Predicted Class: **Severe** or **Proliferative Diabetic Retinopathy**.
  - Badge Color: **Crimson Red / Orange**.
  - Referral Recommendation: Neutral, urgent clinical referral advisory to consult an ophthalmologist for comprehensive evaluation.
* **Pass/Fail Criteria**: Pass if high-severity classification triggers the appropriate specialist referral warning.

---

### Test 3: Grad-CAM Saliency Generation
* **Input**: Any valid retinal fundus image.
* **Procedure**: Run `python src/gradcam.py` or inspect the Streamlit explainability section.
* **Expected Output**:
  - Side-by-side display of the original image and the blended Grad-CAM heatmap.
  - Heatmap dynamically highlights salient regions with warm colors (red, yellow, orange).
  - File `outputs/gradcam.jpg` is generated.
  - Clear caption displayed explaining that the heatmap indicates algorithmic attention, not definitive lesion proof.
* **Pass/Fail Criteria**: Pass if the overlay is correctly blended without crashing and saves to `outputs/gradcam.jpg`.

---

### Test 4: Multiclass Probability Distribution
* **Input**: Any fundus photograph.
* **Procedure**: Complete analysis in Streamlit and observe the horizontal probability chart.
* **Expected Output**:
  - 5 bars displayed representing:
    1. No Diabetic Retinopathy
    2. Mild Diabetic Retinopathy
    3. Moderate Diabetic Retinopathy
    4. Severe Diabetic Retinopathy
    5. Proliferative Diabetic Retinopathy
  - All percentages sum to approximately $100\%$.
* **Pass/Fail Criteria**: Pass if all 5 classes are displayed and probabilities are normalized.

---

### Test 5: Missing / Unuploaded Image
* **Input**: No file selected or uploaded in Streamlit.
* **Procedure**: Open `app.py` in browser without uploading an image.
* **Expected Output**:
  - Informational message: *"Please upload a fundus photograph or choose a sample image from the menu above to begin."*
  - The "Analyze Retinal Image" button remains hidden or inactive.
  - No Python traceback or error exception is shown to the user.
* **Pass/Fail Criteria**: Pass if the UI cleanly guides the user without crashing.

---

### Test 6: Invalid or Non-Image File
* **Input**: Upload a `.txt`, `.pdf`, or corrupt binary file.
* **Procedure**: Attempt to upload via `st.file_uploader`.
* **Expected Output**:
  - Streamlit's file filter immediately disallows or flags invalid file extensions (accepts only `.jpg`, `.jpeg`, `.png`).
  - If a corrupted file with a `.png` extension is uploaded, a friendly notification appears: *"Error reading image file. Please ensure it is an uncorrupted JPG or PNG."*
* **Pass/Fail Criteria**: Pass if no raw Python stack trace is exposed to the user.

---

### Test 7: Missing Model File (`models/dr_model.keras`)
* **Input**: Rename or temporarily delete `models/dr_model.keras`.
* **Procedure**: Refresh Streamlit and attempt to analyze an image.
* **Expected Output**:
  - Sidebar updates to show a warning badge: *"Model Status: dr_model.keras not found."*
  - When the user clicks analyze, an informative banner appears:
    *"Trained model not found at `models/dr_model.keras`! To train the model: run `python src/train.py`."*
* **Pass/Fail Criteria**: Pass if the system handles a missing model file gracefully without application termination.

---

### Test 8: Multiple Image Formats (JPG vs PNG)
* **Input**:
  - Image A: `sample.jpg` (JPEG compression).
  - Image B: `sample.png` (Lossless PNG).
* **Procedure**: Upload each image in succession and run prediction.
* **Expected Output**:
  - Both file formats are ingested without error.
  - Both images are converted to standardized 3-channel RGB arrays.
* **Pass/Fail Criteria**: Pass if predictions and Grad-CAM overlays render identically regardless of file format.

---

### Test 9: Arbitrary Image Dimensions & Aspect Ratios
* **Input**: Retinal images of various dimensions (e.g., $1024 \times 1024$, $1920 \times 1080$, $512 \times 512$).
* **Procedure**: Upload non-224x224 images.
* **Expected Output**:
  - `preprocess_image` resizes input to $(224, 224)$ using bilinear interpolation.
  - Grad-CAM heatmap dynamically rescales back to the original image dimensions for accurate visual overlay.
* **Pass/Fail Criteria**: Pass if dimensions are handled automatically without tensor shape mismatch errors.

---

### Test 10: Streamlit UI Responsiveness & Ethical Warnings
* **Input**: User browses the three tabs (*Retinal Screening & Grad-CAM*, *Clinical Background & AI*, *Model Architecture*).
* **Expected Output**:
  - Fluid tab switching without state loss.
  - Prominently rendered ethical medical disclaimer banner on every tab:
    *"This software is designed for academic research and preliminary screening triaging. It is not an authorized medical diagnostic device..."*
  - Sidebar displays project overview and rural healthcare mission context.
* **Pass/Fail Criteria**: Pass if all sections render correctly and disclaimers are visibly present.

---

## 3. Summary of Test Results

| Test # | Scenario | Component Tested | Status |
| :---: | :--- | :--- | :---: |
| 1 | Normal Retinal Fundus (Class 0) | Pipeline / Class Head | **PASS** |
| 2 | Severe / Proliferative Retinopathy | Referral Logic / Triage | **PASS** |
| 3 | Grad-CAM Heatmap Generation | Explainable AI (`gradcam.py`) | **PASS** |
| 4 | Multiclass Probability Chart | Probability Normalization | **PASS** |
| 5 | Missing Image Handling | UI Error Handling | **PASS** |
| 6 | Invalid / Corrupted File | File Ingestion Shielding | **PASS** |
| 7 | Missing Model Graceful Banner | Resource Fallback | **PASS** |
| 8 | Multiple Formats (JPG/PNG) | Image IO (Pillow) | **PASS** |
| 9 | Dimension Invariance | Preprocessing Resizing | **PASS** |
| 10 | Ethical Disclaimer Visibility | Medical AI Ethics Compliance | **PASS** |
