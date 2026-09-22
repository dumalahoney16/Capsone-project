"""
app.py
==============================================================================
Explainable AI-Based Diabetic Retinopathy Screening
Streamlit Web Application

A student-friendly medical screening decision-support application focused
on potential use in rural and underserved community screening settings.

Features:
  1. Educational Home Section (Diabetic Retinopathy & AI Screening context)
  2. Fundus Image Upload (JPG, JPEG, PNG) or Demo Sample Selection
  3. EfficientNetB0 Deep Learning Severity Prediction & Confidence Score
  4. 5-Class Probability Breakdown Distribution Chart
  5. Explainable AI (Grad-CAM) Visual Heatmap Overlay
  6. Responsible Clinical Referral Advisory
  7. Prominent Ethical and Medical Screening Disclaimers

Usage:
  streamlit run app.py
==============================================================================
"""

import os
import sys
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Explainable AI-Based Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Canonical DR Class Names
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

SEVERITY_COLORS = {
    0: "#28a745",  # Green
    1: "#17a2b8",  # Teal/Cyan
    2: "#ffc107",  # Amber/Yellow
    3: "#fd7e14",  # Orange
    4: "#dc3545"   # Crimson Red
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "dr_model.keras")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


# ==============================================================================
# Cached Model Loader
# ==============================================================================
@st.cache_resource(show_spinner="Loading deep learning model...")
def load_dr_model():
    """Loads the trained Keras model if available."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# ==============================================================================
# Sidebar
# ==============================================================================
def render_sidebar(model_available: bool):
    pass


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_sample_images():
    """Finds any sample images present in dataset/ to allow instant 1-click testing."""
    samples = {}
    folder_mapping = {
        "No_DR": "Class 0: No DR",
        "Mild": "Class 1: Mild DR",
        "Moderate": "Class 2: Moderate DR",
        "Severe": "Class 3: Severe DR",
        "Proliferate_DR": "Class 4: Proliferate DR"
    }
    for folder, label in folder_mapping.items():
        dir_path = os.path.join(DATASET_DIR, folder)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if files:
                samples[f"{label} ({files[0]})"] = os.path.join(dir_path, files[0])
    return samples


def run_prediction_and_gradcam(model, image: Image.Image):
    """Executes model inference and Grad-CAM overlay."""
    from src.predict import preprocess_image
    from src.gradcam import generate_gradcam_overlay

    # Preprocess image
    tensor = preprocess_image(image, target_size=(224, 224))

    # Inference
    raw_probs = model.predict(tensor, verbose=0)[0]
    pred_idx = int(np.argmax(raw_probs))
    confidence = float(raw_probs[pred_idx]) * 100.0
    pred_class_name = CLASS_NAMES[pred_idx]

    prob_dict = {CLASS_NAMES[i]: float(raw_probs[i]) for i in range(len(CLASS_NAMES))}

    # Explainable AI: Grad-CAM
    overlay_img, heatmap = generate_gradcam_overlay(model, tensor, image, class_idx=pred_idx)

    return {
        "pred_idx": pred_idx,
        "class_name": pred_class_name,
        "confidence": confidence,
        "probabilities": prob_dict,
        "overlay": overlay_img,
        "heatmap": heatmap
    }


# ==============================================================================
# Main Application Flow
# ==============================================================================
def main():
    model = load_dr_model()
    render_sidebar(model_available=(model is not None))

    # App Header
    st.title("Explainable AI-Based Diabetic Retinopathy Screening")
    st.markdown(
        "##### EfficientNetB0 Deep Learning with Gradient-Weighted Class Activation Mapping (Grad-CAM)"
    )

    # Mandatory Prominent Disclaimer Banner
    st.info(
        "**Research & Educational Screening Support Tool**: This software is designed "
        "for academic research and preliminary screening triaging. It is **not** an authorized medical "
        "diagnostic device and must not replace clinical examination by a certified ophthalmologist."
    )

    # --------------------------------------------------------------------------
    # Tabs Navigation
    # --------------------------------------------------------------------------
    tab_screening, = st.tabs(
        ["🔍 Retinal Screening & Grad-CAM"]
    )

    # ==========================================================================
    # TAB 1: RETINAL SCREENING & GRAD-CAM
    # ==========================================================================
    with tab_screening:
        st.subheader("1. Retinal Fundus Image Acquisition")
        st.write(
            "Upload a digital retinal fundus photograph (macula/optic disc centered) "
            "or choose an existing sample image to evaluate."
        )

        col_upload, col_sample = st.columns([2, 1])

        uploaded_file = None
        selected_sample_path = None

        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Retinal Fundus Image (JPG, JPEG, PNG)",
                type=["jpg", "jpeg", "png"],
                help="Supports standard fundus camera captures in RGB format."
            )

        with col_sample:
            samples = get_sample_images()
            if samples:
                selected_sample_name = st.selectbox(
                    "Or choose a sample image from dataset:",
                    ["-- Select Sample Image --"] + list(samples.keys())
                )
                if selected_sample_name != "-- Select Sample Image --":
                    selected_sample_path = samples[selected_sample_name]

        # Determine active image
        active_image = None
        image_source_name = ""

        try:
            if uploaded_file is not None:
                active_image = Image.open(uploaded_file).convert("RGB")
                image_source_name = uploaded_file.name
            elif selected_sample_path is not None:
                active_image = Image.open(selected_sample_path).convert("RGB")
                image_source_name = os.path.basename(selected_sample_path)
        except Exception as e:
            st.error(f"Error reading image file: {e}. Please ensure it is an uncorrupted JPG or PNG.")
            return

        if active_image is not None:
            st.divider()
            col_preview, col_action = st.columns([1, 2])

            with col_preview:
                st.image(active_image, caption=f"Selected Image: {image_source_name}", use_container_width=True)

            with col_action:
                st.markdown("### Image Ready for Analysis")
                st.write(f"- Dimensions: **{active_image.width} × {active_image.height}** pixels")
                st.write("- Target Pipeline Resolution: **224 × 224** pixels")
                st.write("- Color Space: **RGB (3 Channels)**")

                analyze_clicked = st.button("🚀 Analyze Retinal Image", type="primary", use_container_width=True)

            # Analysis Execution
            if analyze_clicked:
                if model is None:
                    st.error(
                        "Trained model not found at `models/dr_model.keras`!\n\n"
                        "To train the model:\n"
                        "1. Generate sample images or download APTOS data: `python src/create_sample_data.py`\n"
                        "2. Train the model: `python src/train.py`"
                    )
                    return

                with st.spinner("Executing EfficientNetB0 inference and generating Grad-CAM heatmaps..."):
                    try:
                        results = run_prediction_and_gradcam(model, active_image)
                    except Exception as err:
                        st.error(f"Analysis failed: {err}")
                        return

                st.divider()
                st.subheader("2. Screening Classification & Prediction")

                pred_class = results["class_name"]
                pred_idx = results["pred_idx"]
                confidence = results["confidence"]
                badge_color = SEVERITY_COLORS[pred_idx]

                # Metric Cards
                card_col1, card_col2, card_col3 = st.columns(3)
                with card_col1:
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; border-left: 6px solid {badge_color};
                                    padding: 15px; border-radius: 6px;">
                            <span style="font-size: 13px; color: #6c757d; font-weight: bold;">PREDICTED SEVERITY</span>
                            <h3 style="margin: 5px 0 0 0; color: {badge_color};">{pred_class}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with card_col2:
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; border-left: 6px solid #1f77b4;
                                    padding: 15px; border-radius: 6px;">
                            <span style="font-size: 13px; color: #6c757d; font-weight: bold;">CONFIDENCE SCORE</span>
                            <h3 style="margin: 5px 0 0 0; color: #1f77b4;">{confidence:.1f}%</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with card_col3:
                    stage_text = "Referral Required" if pred_idx > 0 else "Routine Checkup"
                    stage_color = "#dc3545" if pred_idx >= 2 else ("#ffc107" if pred_idx == 1 else "#28a745")
                    st.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; border-left: 6px solid {stage_color};
                                    padding: 15px; border-radius: 6px;">
                            <span style="font-size: 13px; color: #6c757d; font-weight: bold;">SCREENING TRIAGE</span>
                            <h3 style="margin: 5px 0 0 0; color: {stage_color};">{stage_text}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Multiclass Probability Distribution Chart
                st.markdown("#### Multiclass Probability Distribution")
                prob_data = pd.DataFrame({
                    "DR Severity Stage": list(results["probabilities"].keys()),
                    "Probability (%)": [v * 100 for v in results["probabilities"].values()]
                })

                st.bar_chart(
                    data=prob_data.set_index("DR Severity Stage"),
                    y="Probability (%)",
                    color="#1f77b4",
                    use_container_width=True
                )

                # Explainability Section
                st.divider()
                st.subheader("3. Explainable AI: Grad-CAM Saliency Analysis")
                st.markdown(
                    "Grad-CAM (Gradient-weighted Class Activation Mapping) traces gradients backwards from the "
                    "prediction layer to the final convolutional feature layer of EfficientNetB0. "
                    "Warmer colors (**red, orange, yellow**) highlight the anatomical regions that contributed "
                    "most heavily to the model's classification."
                )

                cam_col1, cam_col2 = st.columns(2)
                with cam_col1:
                    st.image(
                        active_image.resize((350, 350)),
                        caption="Original Retinal Fundus Image",
                        use_container_width=True
                    )
                with cam_col2:
                    st.image(
                        results["overlay"].resize((350, 350)),
                        caption=f"Grad-CAM Explanation ({pred_class})",
                        use_container_width=True
                    )

                st.caption(
                    "**Interpretation Notice**: Grad-CAM visualizes algorithmic attention. The highlighted regions "
                    "do not represent definitive proof of medical lesions, microaneurysms, or hemorrhages. "
                    "They show where the convolutional neural network focused when calculating its decision."
                )

                # Clinical Referral Advisory
                st.divider()
                st.subheader("4. Screening Recommendation & Referral Advisory")

                if pred_idx == 0:
                    st.success(
                        "**Screening Observation**: The AI screening pipeline did not identify patterns strongly "
                        "characteristic of Diabetic Retinopathy in this image.\n\n"
                        "**Advisory**: Patients diagnosed with diabetes should continue regular annual dilated "
                        "retinal examinations with an eye-care professional."
                    )
                else:
                    st.warning(
                        f"**Screening Observation**: AI screening suggests possible signs associated with "
                        f"**{pred_class}**.\n\n"
                        "**Clinical Referral Guidance**: Please consult a qualified eye-care professional "
                        "(Ophthalmologist or Optometrist) for comprehensive clinical evaluation, slit-lamp "
                        "biomicroscopy, and dilated fundus examination."
                    )

        else:
            st.info("👆 Please upload a fundus photograph or choose a sample image from the menu above to begin.")


if __name__ == "__main__":
    main()
