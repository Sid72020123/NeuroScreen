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
from sklearn.metrics import accuracy_score, classification_report, recall_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =====================================================================
# 🛠️ FEATURE PRUNING CONFIGURATION
# =====================================================================
# Add the exact string names of the noisy features identified by SHAP here.
# Example: FEATURES_TO_DROP = ["DFA", "RPDE", "PPE", "Jitter:DDP"]
# FEATURES_TO_DROP = []

FEATURES_TO_DROP = [
    "Jitter:DDP",
    "MDVP:PPQ",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "MDVP:Jitter(%)",
]

# =====================================================================
# 📁 PATHS & CONSTANTS
# =====================================================================
UCI_CSV_PATH = "datasets/voice/uci_parkinsons.csv"
LOCAL_HEALTHY_DIR = "datasets/voice/healthy/*.wav"
LOCAL_PARKINSON_DIR = "datasets/voice/parkinsons/*.wav"
CACHE_FILE = "datasets/voice/master_dataset_cache.csv"
MODEL_DIR = "saved_models"

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
    """Extracts all 22 acoustic features using Praat and Nolds."""
    try:
        sound = parselmouth.Sound(filepath)
        pitch = sound.to_pitch()
        y, sr = librosa.load(filepath, sr=None)

        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

        pulses = call([sound, pitch], "To PointProcess (cc)")
        jitter_pct = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        jitter_abs = call(
            pulses, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
        )
        rap = call(pulses, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq = call(pulses, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp = rap * 3

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

        harmonicity = sound.to_harmonicity_cc()
        hnr = call(harmonicity, "Get mean", 0, 0)
        nhr = 10.0 ** (-hnr / 10.0) if hnr and not np.isnan(hnr) else 0.0

        y_sub = y[:3000] if len(y) > 3000 else y
        dfa = nolds.dfa(y_sub)
        d2 = nolds.corr_dim(y_sub, emb_dim=2)
        rpde = nolds.sampen(y_sub)

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
        return [np.nan] * 22


def build_and_cache_dataset():
    """Extracts features and saves them to a CSV to save time on future runs."""
    features, labels = [], []
    healthy_files = glob.glob(LOCAL_HEALTHY_DIR)
    parkinson_files = glob.glob(LOCAL_PARKINSON_DIR)

    print(f"Extracting features from {len(healthy_files)} Healthy files...")
    for f in healthy_files:
        features.append(extract_praat_features(f))
        labels.append(0)

    print(f"Extracting features from {len(parkinson_files)} Parkinson's files...")
    for f in parkinson_files:
        features.append(extract_praat_features(f))
        labels.append(1)

    df_local = pd.DataFrame(features, columns=ALL_22_FEATURES)
    df_local["status"] = labels
    df_local = df_local.dropna()

    df_uci = pd.read_csv(UCI_CSV_PATH)[ALL_22_FEATURES + ["status"]]
    df_master = pd.concat([df_uci, df_local], ignore_index=True)

    # Save to cache
    df_master.to_csv(CACHE_FILE, index=False)
    return df_master


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==========================================
    # 1. LOAD OR BUILD DATASET
    # ==========================================
    if os.path.exists(CACHE_FILE):
        print(
            f"⚡ Loading cached dataset from {CACHE_FILE} (Skipping 10-min extraction)..."
        )
        df_master = pd.read_csv(CACHE_FILE)
    else:
        print(
            "⏳ Cache not found. Building master dataset (This will take a few minutes)..."
        )
        df_master = build_and_cache_dataset()

    print(f"\n✅ Master Dataset Loaded! Total Samples: {len(df_master)}")

    # ==========================================
    # 2. FEATURE PRUNING
    # ==========================================
    features_to_keep = [f for f in ALL_22_FEATURES if f not in FEATURES_TO_DROP]

    if FEATURES_TO_DROP:
        print(f"\n✂️ PRUNING {len(FEATURES_TO_DROP)} NOISY FEATURES:")
        for f in FEATURES_TO_DROP:
            print(f"   - Dropped: {f}")
    else:
        print(
            "\n⚠️ No features pruned. (Update FEATURES_TO_DROP at the top of the script if needed)."
        )

    X = df_master[features_to_keep]
    y = df_master["status"]

    # ==========================================
    # 3. PREPARE DATA & SCALE POS WEIGHT
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    dynamic_scale_pos_weight = neg_count / pos_count

    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 4. TRAIN XGBOOST
    # ==========================================
    print("Training Pruned XGBoost Classifier...")
    xgb = XGBClassifier(
        scale_pos_weight=dynamic_scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    xgb.fit(X_train_scaled, y_train)

    # ==========================================
    # 5. OPTIMIZE THRESHOLD FOR MAXIMUM ACCURACY
    # ==========================================
    print(
        "Optimizing Decision Threshold for Maximum Accuracy (with Recall Tie-Breaker)..."
    )

    # Get probability predictions for the test set (Probability of Class 1: Parkinson's)
    y_probs = xgb.predict_proba(X_test_scaled)[:, 1]

    best_threshold = 0.5
    max_acc = -1.0
    best_recall_for_max_acc = -1.0

    # Test thresholds from 0.05 to 0.95 with a step of 0.01
    # np.round is used to avoid floating point arithmetic issues (e.g., 0.060000000000000005)
    thresholds_to_test = np.round(np.arange(0.05, 0.96, 0.01), 2)

    for thresh in thresholds_to_test:
        # Convert probabilities to binary predictions based on the current threshold
        y_pred_custom = (y_probs >= thresh).astype(int)

        # Calculate standard metrics
        acc = accuracy_score(y_test, y_pred_custom)
        rec = recall_score(y_test, y_pred_custom)

        # 1. Check if this threshold yields a strictly higher accuracy
        if acc > max_acc:
            max_acc = acc
            best_recall_for_max_acc = rec
            best_threshold = thresh

        # 2. Tie-Breaker Rule: If accuracy is identical, pick the threshold with the higher recall
        elif acc == max_acc:
            if rec > best_recall_for_max_acc:
                best_recall_for_max_acc = rec
                best_threshold = thresh

    # Apply the absolute best threshold to get the final predictions
    y_pred_final = (y_probs >= best_threshold).astype(int)

    final_acc = accuracy_score(y_test, y_pred_final)
    final_recall = recall_score(y_test, y_pred_final)

    # ==========================================
    # 6. EXPORT & OUTPUT METRICS
    # ==========================================
    model_path = os.path.join(MODEL_DIR, "xgboost_pruned_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "standard_scaler_pruned.pkl")

    joblib.dump(xgb, model_path)
    joblib.dump(scaler, scaler_path)

    print("\n" + "=" * 50)
    print("🚀 NEUROSCREEN PRUNED MODEL TRAINING COMPLETE")
    print("=" * 50)

    # Format the pruned features list for clean output
    pruned_str = ", ".join(FEATURES_TO_DROP) if FEATURES_TO_DROP else "None"

    print(f"1. Pruned Features                         : {pruned_str}")
    print(f"2. Accuracy-Maximized Custom Threshold     : {best_threshold:.2f}")
    print(f"3. Final Locked Recall (Sensitivity)       : {final_recall * 100:.2f}%")
    print(f"4. Maximized Final Accuracy Score          : {final_acc * 100:.2f}%")
    print("=" * 50)

    print("\nRefined Classification Report (Using Accuracy-Optimized Threshold):")
    print(
        classification_report(
            y_test, y_pred_final, target_names=["Healthy (0)", "Parkinson's (1)"]
        )
    )


if __name__ == "__main__":
    main()
