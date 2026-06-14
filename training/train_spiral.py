"""
train_spiral.py

A complete, standalone script to train a Parkinson's disease screening model 
using spiral drawing images. This script uses TensorFlow/Keras and MobileNetV2 
transfer learning, addressing class imbalance and applying safe data augmentations.

Author: Expert Deep Learning & CV Engineer
Target: NeuroScreen Web App (Flask + TensorFlow)
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
EPOCHS = 30
LEARNING_RATE = 0.0001

# ==========================================
# 1. IMAGE PREPROCESSING PIPELINE
# ==========================================
def preprocess_image(image_path):
    """
    Loads an image, applies adaptive thresholding to remove background noise,
    crops tightly around the spiral, resizes, and formats for MobileNetV2.
    """
    # Load image in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None

    # Apply Adaptive Thresholding to clean up paper color, pen texture, and scanner artifacts
    # This creates a binary image: black lines (0) on a white background (255)
    thresh = cv2.adaptiveThreshold(
        img_gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    # Invert the image to find contours (white lines on black background)
    inv_thresh = cv2.bitwise_not(thresh)

    # Find the outermost contours of the drawing
    contours, _ = cv2.findContours(inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the bounding box that encompasses all contours
        x_min, y_min = img_gray.shape[1], img_gray.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
            
        # Add a small padding to the bounding box (e.g., 5 pixels) to avoid cutting edges
        pad = 5
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(img_gray.shape[1], x_max + pad)
        y_max = min(img_gray.shape[0], y_max + pad)
        
        # Crop tightly around the spiral
        cropped = thresh[y_min:y_max, x_min:x_max]
    else:
        # Fallback if no contours are found
        cropped = thresh

    # Resize to exactly 224x224 pixels
    resized = cv2.resize(cropped, IMG_SIZE)

    # Convert single-channel binary image to 3-channel RGB (MobileNetV2 requirement)
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    # Convert to float32 and apply MobileNetV2 specific preprocessing (scales to [-1, 1])
    img_array = np.array(rgb_img, dtype=np.float32)
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

    # 2. STRATIFIED TRAIN/TEST SPLIT
    # 80% training, 20% testing, maintaining the 27/73 class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 4. ADDRESSING CLASS IMBALANCE
    # Calculate class weights automatically to balance the loss function
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"Computed Class Weights: {class_weights}")

    # 3. SAFE DATA AUGMENTATION (IN-MEMORY ONLY)
    # Strictly avoiding flips, elastic distortions, and shearing
    train_datagen = ImageDataGenerator(
        rotation_range=15,       # Random rotations up to 15 degrees
        zoom_range=0.10,         # Slight zoom up to 10%
        width_shift_range=0.05,  # Minor width translation
        height_shift_range=0.05, # Minor height translation
        horizontal_flip=False,   # PROHIBITED: Spirals have fixed chiral directions
        vertical_flip=False,     # PROHIBITED
        shear_range=0.0,         # PROHIBITED: Can mimic/mask tremors
        fill_mode='constant',
        cval=255                 # Fill with white background
    )
    
    # Note: Test data should NOT be augmented
    test_datagen = ImageDataGenerator()

    # 5. MOBILENETV2 MODEL ARCHITECTURE
    print("Building MobileNetV2 model...")
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Dense layer with L2 regularization to prevent overfitting
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    # High dropout rate
    x = Dropout(0.5)(x)
    # Final output layer (0 = Healthy, 1 = Parkinson's)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 6. COMPILATION, TRAINING, AND CALLBACKS
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping monitoring validation loss
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    print("Starting training...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )

    # 7. CLINICAL EVALUATION METRICS
    print("\n--- Evaluating Model on Unseen Test Set ---")
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Calculate standard metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Handle potential edge cases in confusion matrix shape
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    print(f"\nOverall Accuracy: {acc * 100:.2f}%")
    print(f"Specificity (True Negative Rate): {specificity * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Parkinson\'s (1)']))

    print("\n" + "="*60)
    print("CLINICAL REMINDER FOR NEUROSCREEN:")
    print("In medical screening utilities, high SENSITIVITY (Recall for Class 1) ")
    print("is absolutely critical. A False Negative (missing a Parkinson's patient) ")
    print("is far more dangerous than a False Positive (sending a healthy person ")
    print("for further clinical evaluation). Always prioritize tuning the model ")
    print("to minimize False Negatives.")
    print("="*60 + "\n")

    # 8. MODEL SERIALIZATION
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    model.save(MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()