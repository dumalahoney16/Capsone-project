"""
evaluate.py
==============================================================================
Model Evaluation Module for Diabetic Retinopathy Screening.

Purpose:
--------
Evaluates the trained model on the validation dataset using:
  - Multi-class Confusion Matrix
  - Precision, Recall (Sensitivity), and F1-Score per class
  - Macro-average and Weighted-average metrics
  - Detailed medical screening perspective on why Recall matters

Outputs:
--------
  - Saved Confusion Matrix visualization: `outputs/confusion_matrix.png`
  - Printed comprehensive Classification Report

Usage:
------
    python src/evaluate.py
==============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "dr_model.keras")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

# Canonical class ordering and labels
CLASS_NAMES = [
    "No DR (0)",
    "Mild (1)",
    "Moderate (2)",
    "Severe (3)",
    "Proliferative (4)"
]
RAW_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str):
    """
    Renders and saves an annotated, professionally styled confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1,
        linecolor="#cccccc",
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Diabetic Retinopathy Screening - Confusion Matrix", fontsize=13, pad=12)
    plt.xlabel("Predicted DR Grade", fontsize=11)
    plt.ylabel("Ground Truth DR Grade", fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  [+] Confusion matrix heatmap saved to: {output_path}")


def evaluate():
    print("=" * 75)
    print("DIABETIC RETINOPATHY SCREENING - MODEL EVALUATION")
    print("=" * 75)

    # 1. Check Model Existence
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        print("Please train the model first by running: `python src/train.py`")
        sys.exit(1)

    print(f"Loading trained model from: {MODEL_PATH} ...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.\n")

    # 2. Check Dataset
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    print("Loading validation dataset (20% split) ...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.20,
        subset="validation",
        seed=1337,
        image_size=(224, 224),
        batch_size=16,
        class_names=RAW_CLASSES,
        shuffle=False
    )

    # 3. Extract True Labels and Model Predictions
    print("Running predictions on validation samples...")
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    if len(y_true) == 0:
        print("[ERROR] Validation set is empty! Ensure dataset has sufficient images.")
        sys.exit(1)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        digits=4,
        zero_division=0
    )

    # 5. Print Detailed Medical Evaluation Report
    print("\n" + "-" * 75)
    print("EVALUATION RESULTS")
    print("-" * 75)
    print(f"Overall Validation Accuracy: {acc * 100:.2f}%\n")
    print("Detailed Classification Report:\n")
    print(report)

    # 6. Save Confusion Matrix
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plot_confusion_matrix(cm, CLASS_NAMES, CONFUSION_MATRIX_PATH)

    # 7. Medical AI Interpretation & Discussion
    print("\n" + "=" * 75)
    print("CLINICAL & MEDICAL SCREENING INTERPRETATION:")
    print("=" * 75)
    print("1. Recall (Sensitivity):")
    print("   - In a screening scenario, Recall for severe/proliferative DR is paramount.")
    print("   - A False Negative (missing a patient with sight-threatening retinopathy)")
    print("     can lead to permanent vision loss due to delayed treatment.")
    print("\n2. Precision (Positive Predictive Value):")
    print("   - High precision minimizes False Positives.")
    print("   - In low-resource and rural clinics, false positives overwhelm limited")
    print("     tertiary ophthalmologists with unnecessary referrals.")
    print("\n3. F1-Score:")
    print("   - Harmonic mean of Precision and Recall. Crucial for assessing performance")
    print("     under class imbalance (e.g. fewer Proliferative vs Normal retinas).")
    print("\n4. Performance Integrity:")
    print("   - These metrics reflect experimental screening results on the dataset.")
    print("   - This model is not clinically certified and is intended strictly for")
    print("     educational and research decision support.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    evaluate()
