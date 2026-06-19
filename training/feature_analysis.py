import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =====================================================================
# 📁 CONFIGURATION & PATHS
# =====================================================================
UCI_CSV_PATH = "datasets/voice/uci_parkinsons.csv"
LOCAL_HEALTHY_DIR = "datasets/voice/healthy/*.wav"
LOCAL_PARKINSON_DIR = "datasets/voice/parkinsons/*.wav"
CACHE_FILE = "datasets/voice/master_dataset_cache.csv"
MODEL_DIR = "saved_models"

MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_22feat_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "standard_scaler.pkl")

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

# [Note: To keep this script clean, we assume the cache file exists from your previous runs.
# If not, it will instruct you to run the training script first to generate it.]


def main():
    print("=" * 50)
    print("🔍 NEUROSCREEN: SHAP FEATURE ANALYSIS")
    print("=" * 50)

    # 1. Load the Model and Scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Error: Could not find the trained model or scaler in 'saved_models/'.")
        return

    print("Loading XGBoost Model and StandardScaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. Load the Cached Dataset
    if not os.path.exists(CACHE_FILE):
        print(f"❌ Error: Cache file '{CACHE_FILE}' not found.")
        print("Please run the updated 'train_voice_pruned.py' script first to generate the dataset cache.")
        return

    print("Loading cached dataset...")
    df_master = pd.read_csv(CACHE_FILE)

    X = df_master[ALL_22_FEATURES]
    y = df_master["status"]

    # Recreate the exact Test Set used during training (Random State 42)
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # 3. Generate SHAP Values
    print("\nCalculating SHAP values (Explainable AI)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # 4. Plot and Save
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(12, 8))

    # Generate the plot (show=False prevents it from blocking the terminal)
    shap.summary_plot(shap_values, X_test_scaled, feature_names=ALL_22_FEATURES, show=False)

    plot_filename = "shap_feature_importance.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"✅ Success! High-resolution plot saved as '{plot_filename}'")

    # 5. Console Output Guide
    print("\n" + "=" * 50)
    print("📊 HOW TO READ YOUR SHAP PLOT:")
    print("=" * 50)
    print("1. Y-Axis (Top to Bottom): Features at the top are the MOST important to the model.")
    print("   Features at the very bottom are contributing almost nothing (or adding noise).")
    print("2. X-Axis (Left to Right): Shows the impact on the prediction.")
    print("   Dots pushed to the RIGHT increase the probability of a Parkinson's diagnosis.")
    print("   Dots pushed to the LEFT decrease the probability (Healthy).")
    print("3. Color (Blue to Red): Represents the actual value of the feature.")
    print("   Example: If 'spread1' has RED dots on the RIGHT, it means HIGH spread1 values")
    print("   strongly trigger a Parkinson's prediction.")
    print("\n👉 ACTION ITEM: Look at the bottom 3-5 features on this plot. Add their exact")
    print("string names to the 'FEATURES_TO_DROP' list in 'train_voice_pruned.py'.")
    print("=" * 50)


if __name__ == "__main__":
    main()
