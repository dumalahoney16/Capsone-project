"""
predict.py
==============================================================================
Inference & Prediction Module for Diabetic Retinopathy Severity.

Purpose:
--------
Loads the trained model (`models/dr_model.keras`), preprocesses an input retinal
fundus image, and computes:
  1. The predicted DR severity stage (0 to 4).
  2. The prediction confidence percentage.
  3. Probability distribution across all 5 clinical stages.

Exact Class Names:
  - Class 0: No Diabetic Retinopathy
  - Class 1: Mild Diabetic Retinopathy
  - Class 2: Moderate Diabetic Retinopathy
  - Class 3: Severe Diabetic Retinopathy
  - Class 4: Proliferative Diabetic Retinopathy

Usage:
  - Standalone:
      python src/predict.py --image path/to/retina.jpg
  - Programmatic:
      from src.predict import predict_image
      result = predict_image("path/to/retina.jpg")
==============================================================================
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "dr_model.keras")

# Canonical human-readable class names
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

# Global cache for the loaded model to avoid reloading on every request in Streamlit
_CACHED_MODEL = None


def get_model(model_path: str = MODEL_PATH):
    """
    Loads and caches the trained Keras model for fast subsequent inferences.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                f"Please train the model first by running `python src/train.py`."
            )
        # Load model using modern Keras loader
        _CACHED_MODEL = keras.models.load_model(model_path)
    return _CACHED_MODEL


def preprocess_image(image_input, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses a retinal fundus image for EfficientNetB0:
      - Converts image to RGB (stripping alpha channels or grayscale).
      - Resizes to (224, 224) using high-quality bilinear interpolation.
      - Converts to float32 NumPy array with batch dimension (1, 224, 224, 3).
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input.astype("uint8"))
    else:
        raise ValueError("Unsupported image input type. Provide a file path, PIL Image, or NumPy array.")

    # Ensure 3-channel RGB
    img = img.convert("RGB")
    # Resize to model input size
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert to NumPy array
    img_array = np.array(img, dtype=np.float32)
    # Add batch dimension: (1, 224, 224, 3)
    img_tensor = np.expand_dims(img_array, axis=0)

    return img_tensor


def predict_image(image_input, model_path: str = MODEL_PATH) -> dict:
    """
    Runs model inference on a single retinal image.

    Returns:
      dict containing:
        - 'class_id': int (0 to 4)
        - 'class_name': str (e.g. 'Moderate Diabetic Retinopathy')
        - 'confidence': float (e.g. 87.4)
        - 'probabilities': dict of {class_name: probability_float}
    """
    model = get_model(model_path)
    preprocessed_tensor = preprocess_image(image_input, target_size=(224, 224))

    # Run inference
    raw_predictions = model.predict(preprocessed_tensor, verbose=0)
    probabilities = raw_predictions[0]  # array of length 5

    pred_class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[pred_class_id]) * 100.0
    pred_class_name = CLASS_NAMES[pred_class_id]

    prob_dict = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "class_id": pred_class_id,
        "class_name": pred_class_name,
        "confidence": round(confidence, 2),
        "probabilities": prob_dict,
        "raw_tensor": preprocessed_tensor
    }


def main():
    """
    CLI runner for testing predict.py directly.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Predict Diabetic Retinopathy Grade from Retinal Image")
    parser.add_argument("--image", type=str, help="Path to retinal image")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        # Check if sample image exists
        sample_candidate = os.path.join(BASE_DIR, "dataset", "Moderate", "sample_moderate_001.png")
        if os.path.exists(sample_candidate):
            image_path = sample_candidate
            print(f"No image supplied. Using sample test image: {image_path}")
        else:
            print("Please specify an image to predict: python src/predict.py --image path/to/retina.png")
            sys.exit(1)

    print("=" * 70)
    print("DIABETIC RETINOPATHY PREDICTION")
    print("=" * 70)
    print(f"Input Image: {image_path}")

    try:
        res = predict_image(image_path)
        print("\nPrediction Result:")
        print(f"  Classification : {res['class_name']} (Class {res['class_id']})")
        print(f"  Confidence     : {res['confidence']:.2f}%")
        print("\nClass Probabilities:")
        for name, prob in res["probabilities"].items():
            bar = "#" * int(prob * 30)
            print(f"  - {name:<35}: {prob * 100:>5.2f}% | {bar}")
        print("=" * 70)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
