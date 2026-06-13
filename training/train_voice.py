import os
import glob
import pickle
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================================
# 📁 FOLDER STRUCTURE INSTRUCTIONS
# =====================================================================
# Your folder structure is set up perfectly! It looks like this:
#
# training/
#  ├── train_voice.py                <-- This script
#  └── datasets/
#       └── voice/                   <-- The main dataset folder (DATA_DIR)
#            ├── healthy/            <-- Label 0
#            │    ├── AH_064F_...wav
#            │    ├── AH_114S_...wav
#            │    └── ...
#            └── parkinsons/         <-- Label 1
#                 ├── AH_5456...wav
#                 ├── AH_5456...wav
#                 └── ...
# =====================================================================

DATA_DIR = "datasets/voice"


def extract_features(file_path):
    """
    Extracts Mean Jitter, Mean Shimmer, Pitch STD, ZCR, and first 3 MFCCs using librosa.
    Safely handles empty arrays and division by zero.
    """
    try:
        # Load the audio file. sr=None preserves the original sample rate.
        y, sr = librosa.load(file_path, sr=None)

        # Handle empty audio files safely
        if len(y) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 1. Extract Fundamental Frequency (F0) using Probabilistic YIN (pYIN)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=300, sr=sr, frame_length=2048, hop_length=512
        )

        # 2. Extract RMS Energy (Amplitude) for Shimmer calculation
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        # Ensure arrays are the same length to avoid indexing errors
        min_len = min(len(f0), len(rms), len(voiced_flag))
        f0 = f0[:min_len]
        rms = rms[:min_len]
        voiced_flag = voiced_flag[:min_len]

        # Filter out unvoiced frames (silence or background noise)
        f0_voiced = f0[voiced_flag]
        rms_voiced = rms[voiced_flag]

        # --- Calculate Mean Jitter (local) ---
        if len(f0_voiced) > 1:
            periods = 1.0 / f0_voiced
            mean_period = np.mean(periods)
            if mean_period > 0:
                jitter = np.mean(np.abs(np.diff(periods))) / mean_period
            else:
                jitter = 0.0
        else:
            jitter = 0.0

        # --- Calculate Mean Shimmer (local) ---
        if len(rms_voiced) > 1:
            mean_rms = np.mean(rms_voiced)
            if mean_rms > 0:
                shimmer = np.mean(np.abs(np.diff(rms_voiced))) / mean_rms
            else:
                shimmer = 0.0
        else:
            shimmer = 0.0

        # --- Calculate Pitch Variability (Standard Deviation of F0) ---
        if len(f0_voiced) > 0:
            pitch_std = np.std(f0_voiced)
        else:
            pitch_std = 0.0

        # --- Calculate Zero Crossing Rate (ZCR) ---
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # --- Calculate MFCCs (Mean of the first 3 coefficients) ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
        mfcc_means = np.mean(mfccs, axis=1)

        mfcc1 = mfcc_means[0] if len(mfcc_means) > 0 else 0.0
        mfcc2 = mfcc_means[1] if len(mfcc_means) > 1 else 0.0
        mfcc3 = mfcc_means[2] if len(mfcc_means) > 2 else 0.0

        return jitter, shimmer, pitch_std, zcr, mfcc1, mfcc2, mfcc3

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None, None, None, None


def build_dataset(data_dir):
    """
    Reads the audio files, extracts features, and returns a Pandas DataFrame.
    """
    features = []
    labels = []

    healthy_dir = os.path.join(data_dir, "healthy", "*.wav")
    parkinsons_dir = os.path.join(data_dir, "parkinsons", "*.wav")

    print("Processing 'Healthy' audio files...")
    for file_path in glob.glob(healthy_dir):
        extracted = extract_features(file_path)
        if extracted[0] is not None:
            features.append(extracted)
            labels.append(0)  # 0 = Healthy

    print("Processing 'Parkinson's' audio files...")
    for file_path in glob.glob(parkinsons_dir):
        extracted = extract_features(file_path)
        if extracted[0] is not None:
            features.append(extracted)
            labels.append(1)  # 1 = Parkinson's

    # Create the Pandas DataFrame with the new feature columns
    columns = ["Jitter", "Shimmer", "Pitch_STD", "ZCR", "MFCC_1", "MFCC_2", "MFCC_3"]
    df = pd.DataFrame(features, columns=columns)
    df["Label"] = labels

    return df


def main():
    # 1. Verify the data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: The directory '{DATA_DIR}' was not found.")
        print("Please read the folder structure instructions at the top of the script.")
        return

    # 2. Build the dataset
    print("Extracting acoustic features. This may take a moment...")
    df = build_dataset(DATA_DIR)

    if df.empty:
        print("❌ Error: No valid .wav files found. Please check your data folder.")
        return

    if len(df["Label"].unique()) < 2:
        print(
            "❌ Error: You need BOTH healthy and parkinsons audio files to train the model."
        )
        return

    print("\n✅ Dataset successfully created!")
    print(df.head())

    # 3. Prepare data for training
    feature_columns = [
        "Jitter",
        "Shimmer",
        "Pitch_STD",
        "ZCR",
        "MFCC_1",
        "MFCC_2",
        "MFCC_3",
    ]
    X = df[feature_columns]
    y = df["Label"]

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train the Random Forest Classifier
    print("\nTraining the Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy on Test Set: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Healthy (0)", "Parkinson's (1)"]
        )
    )

    # Print Feature Importances
    print("\n📊 Feature Importances:")
    importances = model.feature_importances_
    for name, importance in zip(feature_columns, importances):
        print(f"  - {name}: {importance * 100:.2f}%")

    # 5. Save the trained model
    model_filename = "voice_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)

    print(f"\n💾 Success! The trained model has been saved as '{model_filename}'.")
    print("You can now load this .pkl file in your Flask app to make predictions!")


if __name__ == "__main__":
    main()
