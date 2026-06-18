"""
train_spiral.py

A complete, standalone script to train a Parkinson's disease screening model
using spiral drawing images. This script uses TensorFlow/Keras and MobileNetV2
transfer learning, addressing class imbalance and applying safe data augmentations.
It features a strict OpenCV preprocessing pipeline to eliminate training-serving skew
and utilizes a Two-Phase Fine-Tuning strategy to maximize accuracy.

Author: Senior Python Machine Learning Engineer
Target: NeuroScreen Web App (Flask + TensorFlow) - ASEP-2 Project
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
BASE_DIR = "datasets/spiral"
HEALTHY_DIR = os.path.join(BASE_DIR, "healthy")
PARKINSONS_DIR = os.path.join(BASE_DIR, "parkinsons")
MODEL_SAVE_DIR = "saved_models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "spiral_model.h5")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PHASE_1_EPOCHS = 15
PHASE_2_EPOCHS = 30


# ==========================================
# 1. STRICT OPENCV PREPROCESSING PIPELINE
# ==========================================
def preprocess_image(image_path):
    """
    Strict preprocessing pipeline to match production exactly.
    Applies Gaussian Blur, Otsu's Thresholding, and formats for MobileNetV2.
    """
    # 1. Read image and convert to grayscale
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None

    # 2. Apply Gaussian Blur to smooth out high-frequency noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Apply Otsu's Thresholding (Inverse)
    # This completely separates the background from the ink.
    # THRESH_BINARY_INV makes the ink white (255) and the background black (0).
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Convert back to 3-channel RGB
    rgb_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # 5. Resize to (224, 224) and apply MobileNetV2 preprocessing
    resized = cv2.resize(rgb_img, IMG_SIZE)
    img_array = np.array(resized, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array)

    return img_preprocessed


def load_dataset():
    """
    Iterates through the dataset directories, processes images, and builds X and y arrays.
    Healthy = 0, Parkinson's = 1.
    """
    X, y = [], []

    # Load Healthy (Class 0)
    if os.path.exists(HEALTHY_DIR):
        for file in os.listdir(HEALTHY_DIR):
            img_path = os.path.join(HEALTHY_DIR, file)
            img = preprocess_image(img_path)
            if img is not None:
                X.append(img)
                y.append(0)
    else:
        print(f"Warning: Directory not found - {HEALTHY_DIR}")

    # Load Parkinson's (Class 1)
    if os.path.exists(PARKINSONS_DIR):
        for file in os.listdir(PARKINSONS_DIR):
            img_path = os.path.join(PARKINSONS_DIR, file)
            img = preprocess_image(img_path)
            if img is not None:
                X.append(img)
                y.append(1)
    else:
        print(f"Warning: Directory not found - {PARKINSONS_DIR}")

    return np.array(X), np.array(y)


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
def main():
    print("Loading and preprocessing dataset...")
    X, y = load_dataset()

    if len(X) == 0:
        print("Error: No images loaded. Please check your dataset directories.")
        return

    print(f"Total images loaded: {len(X)}")
    print(f"Class distribution - Healthy: {np.sum(y==0)}, Parkinson's: {np.sum(y==1)}")

    # 2. DATA SETUP & AUGMENTATION
    # 80/20 Stratified Split to preserve the class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Compute class_weight to penalize minority class errors
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"Computed Class Weights: {class_weights}")

    # Kinematic-Safe Data Augmentation for training data
    # Upgraded to prevent overfitting on the small dataset
    train_datagen = ImageDataGenerator(
        rotation_range=10,  # Only slight rotations
        zoom_range=0.05,  # Very minor scaling
        horizontal_flip=True,  # Enabled for kinematic-safe augmentation
        vertical_flip=True,  # Enabled for kinematic-safe augmentation
        fill_mode="nearest",  # Upgraded fill mode
    )

    # Test data should NOT be augmented (strictly rescale-only/passthrough)
    # Note: Rescaling is already handled by `preprocess_input` in the load_dataset phase
    test_datagen = ImageDataGenerator()

    # ==========================================
    # 3. PHASE 1 TRAINING (WARM-UP)
    # ==========================================
    print("\n--- Starting Phase 1: Warm-up Training ---")

    # Load MobileNetV2 without top classification layer
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Freeze the entire base model
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile with Adam(learning_rate=1e-3)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train for 15 epochs
    history_phase1 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        epochs=PHASE_1_EPOCHS,
        class_weight=class_weights,
        verbose=1,
    )

    # ==========================================
    # 4. PHASE 2 TRAINING (DEEP FINE-TUNING)
    # ==========================================
    print("\n--- Starting Phase 2: Deep Fine-Tuning ---")

    # Unfreeze the base_model
    base_model.trainable = True

    # Iterate through the layers and freeze all EXCEPT the top 50 layers (Upgraded from 20)
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Recompile with a severely reduced learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Setup Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )

    # Train Phase 2 for up to 30 additional epochs
    history_phase2 = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        epochs=PHASE_2_EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # ==========================================
    # 5. EVALUATION & EXPORT
    # ==========================================
    print("\n--- Evaluating Final Model on Unseen Test Set ---")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Calculate standard metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Healthy (0)", "Parkinson's (1)"]
        )
    )

    print("\n" + "=" * 60)
    print("CLINICAL REMINDER FOR NEUROSCREEN:")
    print("In medical screening utilities, high SENSITIVITY (Recall for Class 1) ")
    print("is absolutely critical. A False Negative (missing a Parkinson's patient) ")
    print("is far more dangerous than a False Positive (sending a healthy person ")
    print("for further clinical evaluation). Always prioritize tuning the model ")
    print("to minimize False Negatives.")
    print("=" * 60 + "\n")

    # Save the final optimized model
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    model.save(MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
