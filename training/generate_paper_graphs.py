import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Suppress warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

# =====================================================================
# 📁 CONFIGURATION & PATHS
# =====================================================================
MODEL_PATH = "saved_models/xgboost_pruned_model.pkl"
SCALER_PATH = "saved_models/standard_scaler_pruned.pkl"
CACHE_FILE = "datasets/voice/master_dataset_cache.csv"

# The 5 noisy features identified via SHAP analysis to be dropped
FEATURES_TO_DROP = [
    "Jitter:DDP",
    "MDVP:PPQ",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "MDVP:Jitter(%)",
]

# The locked optimal threshold determined during training
CUSTOM_THRESHOLD = 0.39


def main():
    print("=" * 60)
    print("📊 NEUROSCREEN: GENERATING PUBLICATION-QUALITY GRAPHS")
    print("=" * 60)

    # 1. Load the Model and Scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(
            "❌ Error: Could not find the trained model or scaler in 'saved_models/'."
        )
        return

    print("Loading deployed XGBoost Model and StandardScaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Load the Cached Master Dataset
    if not os.path.exists(CACHE_FILE):
        print(f"❌ Error: Cache file '{CACHE_FILE}' not found.")
        return

    print("Loading dataset and applying feature pruning...")
    df_master = pd.read_csv(CACHE_FILE)

    # Drop the noisy features
    df_pruned = df_master.drop(columns=FEATURES_TO_DROP, errors="ignore")

    # Separate features (X) and target labels (y)
    X = df_pruned.drop(columns=["status"])
    y = df_pruned["status"]

    # 3. Recreate the Exact Test Environment
    print("Recreating the test split and scaling features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Transform X_test using the loaded scaler
    X_test_scaled = scaler.transform(X_test)

    # 4. Generate Predictions and Probabilities
    # Get the probability of the positive class (Parkinson's)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Apply the custom clinical threshold
    y_pred_custom = (y_probs >= CUSTOM_THRESHOLD).astype(int)

    # Set global plotting style for academic papers
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

    # =====================================================================
    # 📈 GRAPH 1: CONFUSION MATRIX
    # =====================================================================
    print("Generating Graph 1: Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred_custom)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 16, "weight": "bold"},
        xticklabels=["Healthy", "Parkinson's"],
        yticklabels=["Healthy", "Parkinson's"],
    )

    plt.title(
        f"Confusion Matrix (Threshold = {CUSTOM_THRESHOLD})",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("True Clinical Diagnosis", fontsize=14, fontweight="bold", labelpad=10)
    plt.xlabel("Model Prediction", fontsize=14, fontweight="bold", labelpad=10)

    plt.tight_layout()
    plt.savefig("paper_confusion_matrix.png", dpi=300)
    plt.close()

    # =====================================================================
    # 📈 GRAPH 2: ROC CURVE
    # =====================================================================
    print("Generating Graph 2: ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="#2ca02c", lw=2.5, label=f"XGBoost ROC (AUC = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.7)

    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=14, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=14, fontweight="bold")
    plt.title(
        "Receiver Operating Characteristic (ROC) Curve",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig("paper_roc_curve.png", dpi=300)
    plt.close()

    # =====================================================================
    # 📈 GRAPH 3: PRECISION-RECALL CURVE
    # =====================================================================
    print("Generating Graph 3: Precision-Recall Curve...")
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_probs)
    ap_score = average_precision_score(y_test, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall_vals,
        precision_vals,
        color="#9467bd",
        lw=2.5,
        label=f"PR Curve (AP = {ap_score:.3f})",
    )

    # Mark the operational point based on our custom threshold
    idx = np.argmin(np.abs(thresholds_pr - CUSTOM_THRESHOLD))
    plt.plot(
        recall_vals[idx],
        precision_vals[idx],
        marker="o",
        markersize=10,
        color="red",
        label=f"Operational Point (Thresh={CUSTOM_THRESHOLD})",
    )

    plt.xlim([-0.02, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=14, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=14, fontweight="bold")
    plt.title("Precision-Recall Curve", fontsize=16, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig("paper_precision_recall.png", dpi=300)
    plt.close()

    # =====================================================================
    # 📈 GRAPH 4: FEATURE IMPORTANCE (TOP 10)
    # =====================================================================
    print("Generating Graph 4: Feature Importance Bar Chart...")
    importances = model.feature_importances_
    feature_names = X.columns

    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(feature_names))

    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")

    plt.title(
        "Top 10 Acoustic Biomarkers (XGBoost Feature Importance)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Relative Importance (Weight)", fontsize=14, fontweight="bold")
    plt.ylabel("Acoustic Feature", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig("paper_feature_importance_native.png", dpi=300)
    plt.close()

    # =====================================================================
    # 📈 GRAPH 5: THRESHOLD VS. PERFORMANCE
    # =====================================================================
    print("Generating Graph 5: Threshold vs. Performance Plot...")
    thresholds_eval = np.linspace(0.0, 1.0, 100)
    acc_scores = []
    prec_scores = []
    rec_scores = []

    for t in thresholds_eval:
        y_pred_t = (y_probs >= t).astype(int)
        acc_scores.append(accuracy_score(y_test, y_pred_t))
        prec_scores.append(precision_score(y_test, y_pred_t, zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred_t))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_eval, acc_scores, label="Accuracy", color="blue", lw=2)
    plt.plot(thresholds_eval, prec_scores, label="Precision", color="green", lw=2)
    plt.plot(
        thresholds_eval, rec_scores, label="Recall (Sensitivity)", color="orange", lw=2
    )

    # Draw vertical line for the optimal custom threshold
    plt.axvline(
        x=CUSTOM_THRESHOLD,
        color="red",
        linestyle="--",
        lw=2,
        label=f"Optimal Threshold ({CUSTOM_THRESHOLD})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Classification Probability Threshold", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title(
        "Model Performance Across Probability Thresholds",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    plt.legend(loc="lower left", fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig("paper_threshold_tuning.png", dpi=300)
    plt.close()

    # =====================================================================
    # 🧮 CALCULATE FINAL METRICS & PRINT MARKDOWN TABLE
    # =====================================================================
    # Extract True Negatives, False Positives, False Negatives, True Positives
    tn, fp, fn, tp = cm.ravel()

    final_accuracy = accuracy_score(y_test, y_pred_custom)
    final_precision = precision_score(y_test, y_pred_custom, zero_division=0)
    final_recall = recall_score(y_test, y_pred_custom)
    final_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    final_f1 = f1_score(y_test, y_pred_custom)

    print("\n" + "=" * 60)
    print("✅ SUCCESS! All 5 high-resolution graphs have been generated.")
    print("=" * 60)
    print("Saved files in current directory:")
    print(" 1. paper_confusion_matrix.png          (300 DPI)")
    print(" 2. paper_roc_curve.png                 (300 DPI)")
    print(" 3. paper_precision_recall.png          (300 DPI)")
    print(" 4. paper_feature_importance_native.png (300 DPI)")
    print(" 5. paper_threshold_tuning.png          (300 DPI)")
    print("\n📋 COPYABLE MARKDOWN TABLE FOR ACADEMIC PAPER:")
    print("-" * 60)

    markdown_table = f"""
| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | {final_accuracy * 100:.2f}% | Overall correctness of the model |
| **Sensitivity (Recall)** | {final_recall * 100:.2f}% | Ability to correctly identify Parkinson's |
| **Specificity** | {final_specificity * 100:.2f}% | Ability to correctly identify Healthy patients |
| **Precision (PPV)** | {final_precision * 100:.2f}% | Accuracy of positive Parkinson's predictions |
| **F1-Score** | {final_f1 * 100:.2f}% | Harmonic mean of Precision and Recall |

*Note: Metrics evaluated on the test set using an optimized clinical probability threshold of {CUSTOM_THRESHOLD}.*
"""
    print(markdown_table.strip())
    print("-" * 60)


if __name__ == "__main__":
    main()
