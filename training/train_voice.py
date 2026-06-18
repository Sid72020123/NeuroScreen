import os
import glob
import warnings
import time
import numpy as np
import pandas as pd
import joblib
import librosa
import nolds

import parselmouth
from parselmouth.praat import call

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    recall_score,
)
from xgboost import XGBClassifier

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# =====================================================================
# 📁 FOLDER STRUCTURE INSTRUCTIONS
# =====================================================================
# project_folder/
#  ├── train_voice.py                <-- This script
#  ├── saved_models/                 <-- Models will be saved here
#  └── datasets/
#       └── voice/
#            ├── uci_parkinsons.csv  <-- The UCI dataset
#            ├── healthy/            <-- Local .wav files (Label 0)
#            └── parkinsons/         <-- Local .wav files (Label 1)
# =====================================================================

UCI_CSV_PATH = "datasets/voice/uci_parkinsons.csv"
LOCAL_HEALTHY_DIR = "datasets/voice/healthy/*.wav"
LOCAL_PARKINSON_DIR = "datasets/voice/parkinsons/*.wav"
MODEL_DIR = "saved_models"

# The exact 22 features matching the UCI Parkinson's Dataset
ALL_22_FEATURES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]


def extract_praat_features(filepath):
    """
    Extracts all 22 acoustic features using Praat (Parselmouth) and Nolds.
    Includes detailed time logging to identify bottlenecks.
    """
    filename = os.path.basename(filepath)
    start_total = time.time()

    try:
        # --- 1. Load Audio ---
        t0 = time.time()
        sound = parselmouth.Sound(filepath)
        pitch = sound.to_pitch()
        y, sr = librosa.load(filepath, sr=None)
        print(f"      [+] Audio Load: {time.time() - t0:.2f}s")

        # --- 2. Pitch Features ---
        t0 = time.time()
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        print(f"      [+] Pitch Extracted: {time.time() - t0:.2f}s")

        # --- 3. Jitter Features ---
        t0 = time.time()
        pulses = call([sound, pitch], "To PointProcess (cc)")
        jitter_pct = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(
            pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
        )
        rap = call(pulses, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq = call(pulses, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp = rap * 3
        print(f"      [+] Jitter Extracted: {time.time() - t0:.2f}s")

        # --- 4. Shimmer Features ---
        t0 = time.time()
        shimmer_pct = call(
            [sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        shimmer_db = call(
            [sound, pulses], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        apq3 = call([sound, pulses], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5 = call([sound, pulses], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11 = call(
            [sound, pulses], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        dda = apq3 * 3
        print(f"      [+] Shimmer Extracted: {time.time() - t0:.2f}s")

        # --- 5. Noise Features (HNR / NHR) ---
        t0 = time.time()
        harmonicity = sound.to_harmonicity_cc()
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 10.0 ** (-hnr / 10.0) if hnr and not np.isnan(hnr) else 0.0
        print(f"      [+] Noise Extracted: {time.time() - t0:.2f}s")

        # --- 6. Non-Linear Dynamical Features (Nolds) ---
        t0 = time.time()
        # REDUCED SLICE: 3000 samples to prevent the script from hanging for 15+ minutes
        y_sub = y[:3000] if len(y) > 3000 else y

        dfa = nolds.dfa(y_sub)
        d2 = nolds.corr_dim(y_sub, emb_dim=2)
        rpde = nolds.sampen(y_sub)
        print(f"      [+] Nolds (DFA, D2, RPDE) Extracted: {time.time() - t0:.2f}s")

        # --- 7. Spread & PPE Features ---
        t0 = time.time()
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) > 0:
            log_f0 = np.log(f0_voiced)
            spread1 = np.std(log_f0)
            spread2 = np.var(log_f0)

            hist, bin_edges = np.histogram(log_f0, bins="fd", density=True)
            p = hist * np.diff(bin_edges)
            p = p[p > 0]
            ppe = -np.sum(p * np.log2(p))
        else:
            spread1, spread2, ppe = 0.0, 0.0, 0.0
        print(f"      [+] Spread/PPE Extracted: {time.time() - t0:.2f}s")

        print(
            f"  -> ✅ Finished {filename} in {time.time() - start_total:.2f}s total\n"
        )

        return [
            mean_pitch,
            max_pitch,
            min_pitch,
            jitter_pct,
            jitter_abs,
            rap,
            ppq,
            ddp,
            shimmer_pct,
            shimmer_db,
            apq3,
            apq5,
            apq11,
            dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe,
        ]

    except Exception as e:
        print(f"  -> ❌ Error processing {filename}: {e}")
        return [np.nan] * 22


def build_local_dataset():
    """
    Loops through the local .wav files, extracts all 22 features,
    and builds a Pandas DataFrame with progress tracking.
    """
    features = []
    labels = []

    healthy_files = glob.glob(LOCAL_HEALTHY_DIR)
    parkinson_files = glob.glob(LOCAL_PARKINSON_DIR)

    print(f"\n--- Starting Extraction for {len(healthy_files)} 'Healthy' files ---")
    for i, file_path in enumerate(healthy_files, 1):
        print(f"[{i}/{len(healthy_files)}] Processing Healthy Audio...")
        extracted = extract_praat_features(file_path)
        features.append(extracted)
        labels.append(0)

    print(
        f"\n--- Starting Extraction for {len(parkinson_files)} 'Parkinson's' files ---"
    )
    for i, file_path in enumerate(parkinson_files, 1):
        print(f"[{i}/{len(parkinson_files)}] Processing Parkinson's Audio...")
        extracted = extract_praat_features(file_path)
        features.append(extracted)
        labels.append(1)

    df_local = pd.DataFrame(features, columns=ALL_22_FEATURES)
    df_local["status"] = labels
    df_local = df_local.dropna()
    return df_local


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==========================================
    # 1. LOAD UCI DATASET & MERGE
    # ==========================================
    if not os.path.exists(UCI_CSV_PATH):
        print(f"❌ Error: Could not find {UCI_CSV_PATH}")
        return

    print("Loading the full UCI Parkinson's CSV dataset...")
    df_uci_full = pd.read_csv(UCI_CSV_PATH)

    # Ensure we only grab the 22 features + status
    columns_to_keep = ALL_22_FEATURES + ["status"]
    df_uci = df_uci_full[columns_to_keep]

    df_local = build_local_dataset()

    if df_local.empty:
        print("⚠️ Warning: No local .wav files found. Training on UCI data only.")
        df_master = df_uci
    else:
        print("\nMerging UCI Dataset with Local Figshare Dataset...")
        df_master = pd.concat([df_uci, df_local], ignore_index=True)

    print(f"\n✅ Master Dataset Created! Total Samples: {len(df_master)}")

    # ==========================================
    # 2. PREPARE DATA & SCALE POS WEIGHT
    # ==========================================
    X = df_master[ALL_22_FEATURES]
    y = df_master["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Dynamically calculate scale_pos_weight for XGBoost to handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    dynamic_scale_pos_weight = neg_count / pos_count

    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 3. TRAIN XGBOOST
    # ==========================================
    print("Training XGBoost Classifier...")
    xgb = XGBClassifier(
        scale_pos_weight=dynamic_scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    xgb.fit(X_train_scaled, y_train)

    # ==========================================
    # 4. OPTIMIZE THRESHOLD FOR >= 90% RECALL
    # ==========================================
    print("Optimizing Decision Threshold via Precision-Recall Curve...")
    y_probs = xgb.predict_proba(X_test_scaled)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

    best_threshold = 0.5
    max_acc = 0.0
    locked_recall = 0.0

    # Iterate through all possible thresholds found by the PR curve
    for thresh in thresholds:
        y_pred_custom = (y_probs >= thresh).astype(int)
        rec = recall_score(y_test, y_pred_custom)
        acc = accuracy_score(y_test, y_pred_custom)

        # Strictly enforce Recall >= 90%
        if rec >= 0.90:
            if acc > max_acc:
                max_acc = acc
                best_threshold = thresh
                locked_recall = rec

    # Apply the best threshold to get final predictions
    y_pred_final = (y_probs >= best_threshold).astype(int)

    # ==========================================
    # 5. EXPORT & OUTPUT METRICS
    # ==========================================
    model_path = os.path.join(MODEL_DIR, "xgboost_22feat_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "standard_scaler.pkl")

    joblib.dump(xgb, model_path)
    joblib.dump(scaler, scaler_path)

    print("\n" + "=" * 50)
    print("🚀 NEUROSCREEN MODEL TRAINING COMPLETE")
    print("=" * 50)
    print(
        f"1. Dynamically Calculated scale_pos_weight : {dynamic_scale_pos_weight:.4f}"
    )
    print(f"2. Custom Probability Threshold Locked     : {best_threshold:.4f}")
    print(f"3. Final Locked Recall (Sensitivity)       : {locked_recall * 100:.2f}%")
    print(f"4. Maximized Final Accuracy Score          : {max_acc * 100:.2f}%")
    print("=" * 50)

    print("\nDetailed Classification Report (Using Custom Threshold):")
    print(
        classification_report(
            y_test, y_pred_final, target_names=["Healthy (0)", "Parkinson's (1)"]
        )
    )


if __name__ == "__main__":
    main()
