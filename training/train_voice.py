import os
import glob
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Suppress librosa and XGBoost warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

# =====================================================================
# 📁 FOLDER STRUCTURE INSTRUCTIONS
# =====================================================================
# training/
#  ├── train_voice.py                <-- This script
#  ├── saved_models/                 <-- Models will be saved here
#  └── datasets/
#       └── voice/
#            ├── healthy/            <-- Label 0
#            └── parkinsons/         <-- Label 1
# =====================================================================

DATA_DIR = "datasets/voice"
MODEL_DIR = "saved_models"

# Define the 33 Feature Names
FEATURE_NAMES = (
    [
        "Jitter",
        "Shimmer",
        "Pitch_STD",
        "Spectral_Centroid",
        "Spectral_Bandwidth",
        "Spectral_Rolloff",
        "RMS_Energy",
    ]
    + [f"MFCC_{i}_Mean" for i in range(1, 14)]
    + [f"MFCC_{i}_Var" for i in range(1, 14)]
)


def load_audio_data(data_dir):
    """
    Loads raw audio arrays and sample rates into memory.
    """
    audio_data = []
    labels = []

    healthy_dir = os.path.join(data_dir, "healthy", "*.wav")
    parkinsons_dir = os.path.join(data_dir, "parkinsons", "*.wav")

    print("Loading 'Healthy' audio files into memory...")
    for file_path in glob.glob(healthy_dir):
        y, sr = librosa.load(file_path, sr=None)
        audio_data.append((y, sr))
        labels.append(0)  # 0 = Healthy

    print("Loading 'Parkinson's' audio files into memory...")
    for file_path in glob.glob(parkinsons_dir):
        y, sr = librosa.load(file_path, sr=None)
        audio_data.append((y, sr))
        labels.append(1)  # 1 = Parkinson's

    return audio_data, labels


def augment_audio(y, sr):
    """
    Generates 3 augmented variations of the input audio array.
    """
    # 1. Add Light White Noise
    noise_amp = 0.005 * np.random.uniform() * np.amax(y)
    y_noise = y + noise_amp * np.random.normal(size=y.shape[0])

    # 2. Pitch Shift (Up by 2 steps)
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)

    # 3. Time Stretch (Speed up by 10%)
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)

    return [y_noise, y_pitch, y_stretch]


def extract_features(y, sr):
    """
    Extracts 33 acoustic features from an audio array.
    """
    if len(y) == 0:
        return [0.0] * len(FEATURE_NAMES)

    try:
        # --- 1. Pitch & Amplitude Features ---
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=300, sr=sr, frame_length=2048, hop_length=512
        )
        rms_array = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        min_len = min(len(f0), len(rms_array), len(voiced_flag))
        f0 = f0[:min_len]
        rms_array = rms_array[:min_len]
        voiced_flag = voiced_flag[:min_len]

        f0_voiced = f0[voiced_flag]
        rms_voiced = rms_array[voiced_flag]

        # Jitter
        if len(f0_voiced) > 1:
            periods = 1.0 / f0_voiced
            mean_period = np.mean(periods)
            jitter = (
                np.mean(np.abs(np.diff(periods))) / mean_period
                if mean_period > 0
                else 0.0
            )
        else:
            jitter = 0.0

        # Shimmer
        if len(rms_voiced) > 1:
            mean_rms = np.mean(rms_voiced)
            shimmer = (
                np.mean(np.abs(np.diff(rms_voiced))) / mean_rms if mean_rms > 0 else 0.0
            )
        else:
            shimmer = 0.0

        # Pitch STD
        pitch_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0.0

        # --- 2. Spectral Features ---
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        rms_energy = np.mean(librosa.feature.rms(y=y)[0])

        # --- 3. MFCCs (13 Mean, 13 Variance) ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        mfcc_var = np.var(mfccs, axis=1).tolist()

        return (
            [jitter, shimmer, pitch_std, spec_cent, spec_bw, spec_rolloff, rms_energy]
            + mfcc_mean
            + mfcc_var
        )

    except Exception as e:
        return [0.0] * len(FEATURE_NAMES)


def main():
    # 1. Verify directories
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: The directory '{DATA_DIR}' was not found.")
        return
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Load Raw Audio Data
    audio_data, labels = load_audio_data(DATA_DIR)
    if len(labels) < 2:
        print("❌ Error: Not enough audio files found.")
        return

    # 3. Train/Test Split (BEFORE Augmentation to prevent data leakage)
    print("\nSplitting data into Train and Test sets...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        audio_data, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 4. Augment Training Data Only
    print("Applying Data Augmentation to Training Set (Multiplying data by 4x)...")
    X_train_aug = []
    y_train_aug = []

    for (y, sr), label in zip(X_train_raw, y_train_raw):
        # Keep original
        X_train_aug.append((y, sr))
        y_train_aug.append(label)

        # Generate and append augmentations
        augs = augment_audio(y, sr)
        for aug_y in augs:
            X_train_aug.append((aug_y, sr))
            y_train_aug.append(label)

    # 5. Extract Features
    print(
        f"\nExtracting 33 features for {len(X_train_aug)} Training samples (This will take a few minutes)..."
    )
    X_train_feat = [extract_features(y, sr) for y, sr in X_train_aug]

    print(f"Extracting 33 features for {len(X_test_raw)} Testing samples...")
    X_test_feat = [extract_features(y, sr) for y, sr in X_test_raw]

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train_feat, columns=FEATURE_NAMES)
    X_test_df = pd.DataFrame(X_test_feat, columns=FEATURE_NAMES)

    # 6. Scale and SMOTE
    print("\nApplying StandardScaler and SMOTE...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_scaled, y_train_aug
    )

    # 7. XGBoost & Hyperparameter Tuning
    print("\nRunning GridSearchCV to find the best XGBoost parameters...")
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    grid = GridSearchCV(xgb, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_resampled, y_train_resampled)

    best_model = grid.best_estimator_
    print(f"✅ Best Parameters Found: {grid.best_params_}")

    # 8. Evaluation on Untouched Test Set
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test_raw, y_pred)

    print(f"\n🎯 Final Model Accuracy on Untouched Test Set: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test_raw, y_pred, target_names=["Healthy (0)", "Parkinson's (1)"]
        )
    )

    # 9. Plot Feature Importances
    print("\n📊 Generating Feature Importance Plot...")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot top 15 features
    top_n = 15
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Acoustic Biomarkers for Parkinson's Detection")
    plt.barh(
        range(top_n), importances[indices][:top_n], align="center", color="skyblue"
    )
    plt.yticks(range(top_n), [FEATURE_NAMES[i] for i in indices[:top_n]])
    plt.gca().invert_yaxis()
    plt.xlabel("Relative Importance")
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, "feature_importances.png")
    plt.savefig(plot_path)
    print(f"Saved feature importance chart to '{plot_path}'")

    # 10. Save the Model and Scaler
    model_path = os.path.join(MODEL_DIR, "xgboost_voice_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "standard_scaler.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Success! Model saved to '{model_path}'")
    print(f"💾 Success! Scaler saved to '{scaler_path}'")
    print(
        "In your Flask app, remember to load BOTH the scaler and the model to make predictions!"
    )


if __name__ == "__main__":
    main()
