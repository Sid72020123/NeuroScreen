import os
import glob
import warnings
import numpy as np
import pandas as pd
import joblib

import parselmouth
from parselmouth.praat import call

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# =====================================================================
# 📁 FOLDER STRUCTURE INSTRUCTIONS
# =====================================================================
# Your folder structure must look exactly like this:
#
# project_folder/
#  ├── train_combined_voice.py       <-- This script
#  ├── saved_models/                 <-- Models will be saved here
#  └── dataset/
#       ├── uci_parkinsons.csv       <-- The UCI dataset
#       └── figshare/
#            ├── healthy/            <-- Local .wav files (Label 0)
#            └── parkinson/          <-- Local .wav files (Label 1)
# =====================================================================

UCI_CSV_PATH = "datasets/voice/uci_parkinsons.csv"
LOCAL_HEALTHY_DIR = "datasets/voice/healthy/*.wav"
LOCAL_PARKINSON_DIR = "datasets/voice/parkinsons/*.wav"
MODEL_DIR = "saved_models"

# The exact 7 core features we are keeping to match the UCI dataset
CORE_FEATURES = [
    "MDVP:Fo(Hz)",  # Mean Pitch
    "MDVP:Fhi(Hz)",  # Max Pitch
    "MDVP:Flo(Hz)",  # Min Pitch
    "MDVP:Jitter(%)",  # Local Jitter
    "MDVP:Shimmer",  # Local Shimmer
    "NHR",  # Noise-to-Harmonics Ratio
    "HNR",  # Harmonics-to-Noise Ratio
]


def extract_praat_features(filepath):
    """
    Extracts the 7 core acoustic features using Praat (Parselmouth)
    to perfectly match the mathematical calculations of the UCI dataset.
    """
    try:
        # Load audio into Parselmouth Sound object
        sound = parselmouth.Sound(filepath)
        pitch = sound.to_pitch()

        # 1. Pitch Features (Mean, Max, Min)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

        # Create a PointProcess to calculate Jitter and Shimmer
        pulses = call([sound, pitch], "To PointProcess (cc)")

        # 2. Jitter & Shimmer
        # Note: UCI's "MDVP:Jitter(%)" is actually stored as a decimal fraction (e.g., 0.007),
        # which matches Praat's default output perfectly.
        jitter = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call(
            [sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        # 3. Harmonics-to-Noise (HNR) and Noise-to-Harmonics (NHR)
        harmonicity = sound.to_harmonicity_cc()
        hnr = call(harmonicity, "Get mean", 0, 0)

        # Praat calculates HNR in dB. NHR is the inverse ratio: 10^(-HNR/10)
        if hnr and not np.isnan(hnr):
            nhr = 10.0 ** (-hnr / 10.0)
        else:
            nhr = 0.0

        return [mean_pitch, max_pitch, min_pitch, jitter, shimmer, nhr, hnr]

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return [np.nan] * 7


def build_local_dataset():
    """
    Loops through the local .wav files, extracts Praat features,
    and builds a Pandas DataFrame.
    """
    features = []
    labels = []

    print("Extracting Praat features from local 'Healthy' audio files...")
    for file_path in glob.glob(LOCAL_HEALTHY_DIR):
        extracted = extract_praat_features(file_path)
        features.append(extracted)
        labels.append(0)  # 0 = Healthy

    print("Extracting Praat features from local 'Parkinson's' audio files...")
    for file_path in glob.glob(LOCAL_PARKINSON_DIR):
        extracted = extract_praat_features(file_path)
        features.append(extracted)
        labels.append(1)  # 1 = Parkinson's

    # Create DataFrame with exact UCI column names
    df_local = pd.DataFrame(features, columns=CORE_FEATURES)
    df_local["status"] = labels

    # Drop any rows where Praat failed to extract pitch (NaNs)
    df_local = df_local.dropna()
    return df_local


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==========================================
    # 1. LOAD AND PRUNE THE UCI DATASET
    # ==========================================
    if not os.path.exists(UCI_CSV_PATH):
        print(f"❌ Error: Could not find {UCI_CSV_PATH}")
        return

    print("Loading and pruning the UCI Parkinson's CSV dataset...")
    df_uci_full = pd.read_csv(UCI_CSV_PATH)

    # Keep only the 7 core features + the status label
    columns_to_keep = CORE_FEATURES + ["status"]
    df_uci = df_uci_full[columns_to_keep]

    # ==========================================
    # 2. EXTRACT LOCAL DATA & MERGE
    # ==========================================
    df_local = build_local_dataset()

    if df_local.empty:
        print("⚠️ Warning: No local .wav files found. Training on UCI data only.")
        df_master = df_uci
    else:
        print("\nMerging UCI Dataset with Local Figshare Dataset...")
        df_master = pd.concat([df_uci, df_local], ignore_index=True)

    print(f"\n✅ Master Dataset Created! Total Samples: {len(df_master)}")
    print(df_master.head())

    # ==========================================
    # 3. PREPARE DATA FOR TRAINING
    # ==========================================
    X = df_master[CORE_FEATURES]
    y = df_master["status"]

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply StandardScaler
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 4. TRAIN XGBOOST WITH GRIDSEARCHCV
    # ==========================================
    print("Running GridSearchCV to tune XGBoost parameters...")
    xgb = XGBClassifier(random_state=42, eval_metric="logloss")

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
    }

    grid = GridSearchCV(xgb, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    best_model = grid.best_estimator_
    print(f"✅ Best Parameters Found: {grid.best_params_}")

    # ==========================================
    # 5. EVALUATION & EXPORT
    # ==========================================
    print("\nEvaluating the tuned model on the Test Set...")
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Final Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Healthy (0)", "Parkinson's (1)"]
        )
    )

    # Save the Model and Scaler
    model_path = os.path.join(MODEL_DIR, "xgboost_combined_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "standard_scaler.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Success! Model saved to '{model_path}'")
    print(f"💾 Success! Scaler saved to '{scaler_path}'")


if __name__ == "__main__":
    main()
