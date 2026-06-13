import os
import glob
import pickle
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# =====================================================================
# 📁 FOLDER STRUCTURE INSTRUCTIONS
# =====================================================================
# To use this script, organize your .wav audio files into two folders
# based on their class (Healthy vs. Parkinson's).
#
# Create a main directory named "data" in the exact same folder as
# this script. Inside "data", create "healthy" and "parkinsons".
#
# Your folder structure MUST look exactly like this:
#
# project_folder/
#  ├── train_voice.py                <-- This script
#  └── data/                         <-- The main dataset folder
#       ├── healthy/                 <-- Label 0
#       │    ├── healthy_1.wav
#       │    ├── healthy_2.wav
#       │    └── ...
#       └── parkinsons/              <-- Label 1
#            ├── parkinsons_1.wav
#            ├── parkinsons_2.wav
#            └── ...
# =====================================================================

DATA_DIR = "datasets/voice"


def extract_jitter_shimmer(file_path):
    """
    Extracts Mean Jitter (local) and Mean Shimmer (local) using librosa.
    """
    try:
        # Load the audio file. sr=None preserves the original sample rate.
        y, sr = librosa.load(file_path, sr=None)

        # 1. Extract Fundamental Frequency (F0) using Probabilistic YIN (pYIN)
        # pYIN is highly accurate for voice. It returns F0, a boolean array
        # of voiced frames, and voiced probabilities.
        # fmin=50 and fmax=300 Hz covers the typical human vocal pitch range.
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
        # Jitter is the variation in pitch periods (Time = 1 / Frequency)
        if len(f0_voiced) > 1:
            periods = 1.0 / f0_voiced
            mean_period = np.mean(periods)
            if mean_period > 0:
                # Mean absolute difference of consecutive periods / mean period
                jitter = np.mean(np.abs(np.diff(periods))) / mean_period
            else:
                jitter = 0.0
        else:
            jitter = 0.0

        # --- Calculate Mean Shimmer (local) ---
        # Shimmer is the variation in amplitude (loudness)
        if len(rms_voiced) > 1:
            mean_rms = np.mean(rms_voiced)
            if mean_rms > 0:
                # Mean absolute difference of consecutive amplitudes / mean amplitude
                shimmer = np.mean(np.abs(np.diff(rms_voiced))) / mean_rms
            else:
                shimmer = 0.0
        else:
            shimmer = 0.0

        return jitter, shimmer

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


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
        jitter, shimmer = extract_jitter_shimmer(file_path)
        if jitter is not None and shimmer is not None:
            features.append([jitter, shimmer])
            labels.append(0)  # 0 = Healthy

    print("Processing 'Parkinson's' audio files...")
    for file_path in glob.glob(parkinsons_dir):
        jitter, shimmer = extract_jitter_shimmer(file_path)
        if jitter is not None and shimmer is not None:
            features.append([jitter, shimmer])
            labels.append(1)  # 1 = Parkinson's

    # Create the Pandas DataFrame
    df = pd.DataFrame(features, columns=["Jitter", "Shimmer"])
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
    X = df[["Jitter", "Shimmer"]]
    y = df["Label"]

    # Split into training and testing sets (80% train, 20% test)
    # Note: If you are testing with a tiny dataset (< 5 files), remove 'stratify=y'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train the Support Vector Classifier (SVC)
    print("\nTraining the Support Vector Classifier (SVC)...")

    # probability=True is required so we can predict confidence percentages later
    model = SVC(kernel="linear", probability=True, random_state=42)
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

    # 5. Save the trained model
    model_filename = "voice_model.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)

    print(f"\n💾 Success! The trained model has been saved as '{model_filename}'.")
    print(
        "You can now load this .pkl file in a separate script to make predictions on new voices!"
    )


if __name__ == "__main__":
    main()
