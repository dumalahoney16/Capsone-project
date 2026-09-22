"""
train.py
==============================================================================
Transfer Learning Training Pipeline for Diabetic Retinopathy Classification.

Architecture:
    Input (224x224x3)
    -> Augmentation (Flip, Rotation, Zoom)
    -> EfficientNetB0 (ImageNet Pretrained Feature Extractor)
    -> Global Average Pooling 2D
    -> Dropout (0.3)
    -> Dense (128, ReLU)
    -> Dropout (0.2)
    -> Dense (5, Softmax)

Key Features:
    - 2-Phase Training: Feature extraction (frozen base) followed by fine-tuning.
    - Sparse Categorical Cross-Entropy loss for memory efficiency.
    - Real-time callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.
    - Saves model in modern Keras format: `models/dr_model.keras`.
    - Generates and saves training vs. validation accuracy and loss plots in `outputs/`.

Usage:
    python src/train.py
==============================================================================
"""

import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================================================
# Configuration & Hyperparameters
# ==============================================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 5
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
BASE_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "dr_model.keras")
ACCURACY_PLOT_PATH = os.path.join(OUTPUTS_DIR, "training_accuracy.png")
LOSS_PLOT_PATH = os.path.join(OUTPUTS_DIR, "training_loss.png")

# Canonical class ordering mapping to severity
EXPECTED_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def inspect_dataset(dataset_dir: str):
    """
    Checks the dataset folder, counts images per class, and provides
    insights on medical class imbalance.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Inspecting Dataset & Class Distribution")
    print("=" * 70)

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        print("Please create the folder or run `python src/create_sample_data.py` first.")
        sys.exit(1)

    total_images = 0
    class_counts = {}

    for cls in EXPECTED_CLASSES:
        folder_path = os.path.join(dataset_dir, cls)
        if os.path.exists(folder_path):
            valid_exts = (".png", ".jpg", ".jpeg")
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
            count = len(files)
            class_counts[cls] = count
            total_images += count
            print(f"  - {cls:<18}: {count:>4} images")
        else:
            class_counts[cls] = 0
            print(f"  - {cls:<18}: [Directory Missing]")

    print(f"\nTotal Retinal Images Found: {total_images}")

    if total_images == 0:
        print("\n[ALERT] No images found in dataset folders!")
        print("To quickly test this project, run the sample generator:")
        print("    python src/create_sample_data.py")
        print("Or place your APTOS 2019 dataset images into the corresponding dataset/ folders.")
        sys.exit(1)

    # Educational commentary on class imbalance
    print("\n[NOTE ON CLASS IMBALANCE IN MEDICAL SCREENING]:")
    print("In clinical populations, normal retinas (Class 0: No DR) typically outnumber")
    print("severe cases. High overall accuracy can be misleading if a model simply")
    print("predicts the majority class. Therefore, evaluation metrics like Recall (Sensitivity)")
    print("and the Confusion Matrix are essential to ensure diseased cases are not missed.")
    print("=" * 70 + "\n")

    return class_counts


def load_datasets(dataset_dir: str):
    """
    Loads training (80%) and validation (20%) datasets from directory
    using tf.keras.utils.image_dataset_from_directory.
    """
    print("STEP 2: Loading Datasets (80% Train, 20% Validation)...")

    # Set random seed for reproducible train/val splits
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="training",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=EXPECTED_CLASSES,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.20,
        subset="validation",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=EXPECTED_CLASSES,
        shuffle=False
    )

    print(f"  Detected Class Names: {train_ds.class_names}")

    # Optimize dataset loading using prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_data_augmentation_pipeline():
    """
    Constructs data augmentation layers to prevent overfitting
    and simulate real-world fundus imaging variability:
      - Random horizontal and vertical flips
      - Slight rotations (simulating varying head positions during fundus imaging)
      - Mild zoom (simulating camera distance differences)
    """
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08, fill_mode="constant"),
            layers.RandomZoom(height_factor=0.08, width_factor=0.08, fill_mode="constant"),
        ],
        name="data_augmentation"
    )
    return augmentation


def build_dr_model():
    """
    Builds the complete Transfer Learning model using EfficientNetB0:
      1. Augmentation layer
      2. EfficientNetB0 base feature extractor (weights pretrained on ImageNet)
      3. Custom Classification Head with Dropout and Dense layers
    """
    print("\nSTEP 3: Building EfficientNetB0 Transfer Learning Architecture...")

    # Input layer
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")

    # Apply data augmentation
    aug_layer = build_data_augmentation_pipeline()
    x = aug_layer(inputs)

    # Note: EfficientNet models in Keras have built-in normalization layers
    # that expect inputs in range [0, 255], so no manual rescale 1/255 is needed.
    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # Initially freeze all layers in the base model
    base_model.trainable = False

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="dr_predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DiabeticRetinopathy_EfficientNetB0")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"  Model successfully constructed. Total Parameters: {model.count_params():,}")
    return model, base_model


def plot_and_save_curves(history1, history2=None):
    """
    Plots training and validation accuracy and loss curves,
    saving them as PNG files in the outputs/ directory.
    """
    print("\nSTEP 5: Generating and Saving Training Curves...")

    acc = history1.history.get("accuracy", [])
    val_acc = history1.history.get("val_accuracy", [])
    loss = history1.history.get("loss", [])
    val_loss = history1.history.get("val_loss", [])

    if history2 is not None:
        acc += history2.history.get("accuracy", [])
        val_acc += history2.history.get("val_accuracy", [])
        loss += history2.history.get("loss", [])
        val_loss += history2.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    # 1. Training & Validation Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="s", color="#ff7f0e", linewidth=2)
    plt.title("Diabetic Retinopathy Model - Training vs Validation Accuracy", fontsize=13, pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=200)
    plt.close()
    print(f"  [+] Accuracy curve saved to: {ACCURACY_PLOT_PATH}")

    # 2. Training & Validation Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Training Loss", marker="o", color="#d62728", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="s", color="#2ca02c", linewidth=2)
    plt.title("Diabetic Retinopathy Model - Training vs Validation Loss", fontsize=13, pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (Sparse Categorical Cross-Entropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=200)
    plt.close()
    print(f"  [+] Loss curve saved to: {LOSS_PLOT_PATH}")


def train_model():
    """
    Main training routine coordinating inspection, loading, multi-stage
    training, callbacks, and visualization.
    """
    # 1. Inspect data
    inspect_dataset(DATASET_DIR)

    # 2. Load datasets
    train_ds, val_ds = load_datasets(DATASET_DIR)

    # 3. Build model
    model, base_model = build_dr_model()

    # Callbacks for robust training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # 4. Phase 1: Train classification head with frozen base
    print("\n" + "=" * 70)
    print("STEP 4 (Phase 1): Training Classification Head (Base Frozen)")
    print(f"Epochs: {INITIAL_EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {BASE_LEARNING_RATE}")
    print("=" * 70)

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=callbacks
    )

    # 5. Phase 2 (Optional Fine-Tuning): Unfreeze top layers of EfficientNetB0
    print("\n" + "=" * 70)
    print("STEP 4 (Phase 2): Fine-Tuning Top Layers of EfficientNetB0")
    print(f"Unfreezing upper layers with reduced LR: {FINE_TUNE_LEARNING_RATE}")
    print("=" * 70)

    base_model.trainable = True
    # Freeze all layers except the top 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history_phase1.epoch),
        callbacks=callbacks
    )

    # Ensure best model is saved at the target path
    model.save(MODEL_SAVE_PATH)
    print(f"\n[+] Best Model Successfully Saved to: {MODEL_SAVE_PATH}")

    # Plot metrics
    plot_and_save_curves(history_phase1, history_phase2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Trained model saved at : {MODEL_SAVE_PATH}")
    print("Next step: Run `python src/evaluate.py` to evaluate the model.")
    print("=" * 70)


if __name__ == "__main__":
    train_model()
