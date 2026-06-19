# NeuroScreen: A Multi-Modal AI-Powered Platform for Neurological Stress Screening

## 1. Abstract

This paper details the architecture and implementation of NeuroScreen, a multi-modal, non-invasive screening tool designed for the early detection of neurological stress indicators, particularly those associated with neurodegenerative conditions like Parkinson's disease. The system integrates two distinct machine learning models to analyze vocal biomarkers and fine motor control, providing a holistic, data-driven risk assessment. The voice analysis module employs an XGBoost classifier trained on clinically relevant acoustic features extracted via the Praat engine, while the motor skills module uses a fine-tuned MobileNetV2 deep learning model to analyze spiral drawings. The platform is delivered as a Flask web application, featuring comprehensive patient profiling, medication phase tracking, clinician portals, and explainable AI (XAI) components to ensure transparency and interpretability of the results.

---

## 2. Methodology

The project's methodology is bifurcated into two primary analytical streams: voice analysis and motor kinematics (spiral) analysis. Each stream utilizes distinct datasets, feature extraction techniques, and model architectures.

### 2.1. Datasets

#### 2.1.1. Voice Analysis Datasets

- **UCI Parkinson's Dataset:** This foundational dataset consists of 195 samples with 22 pre-extracted acoustic features per voice recording. It served as the primary source for training the voice classification model.
- **Figshare 2023 (Prior et al.):** This dataset contains raw `.wav` audio files from both healthy and Parkinson's-diagnosed individuals. These files were used to augment the training data and, critically, to validate that our feature extraction pipeline (`extract_praat_features`) could replicate the feature space of the UCI dataset from raw audio.

#### 2.1.2. Spiral Drawing Datasets

A combined dataset of **734 images** was aggregated from multiple public sources to train the spiral analysis model.
- **HandPD & NewHandPD (UNESP Brazil):** Provided the core set of spiral and wave drawings from healthy and patient cohorts.
- **Kaggle Parkinson's Drawings (kmader):** Contributed an additional 102 spiral images, further diversifying the training set.

### 2.2. Voice Analysis Model (XGBoost Classifier)

#### 2.2.1. Feature Extraction (`extract_praat_features`)

A custom feature extraction function was developed to process raw `.wav` files using the `parselmouth` library (Praat) and `nolds`.
- For each audio file, the function calculates non-linear acoustic features. Through SHAP analysis, 5 noisy features were pruned, resulting in a highly optimized 17-feature input vector. High-impact features include: `spread2`, `D2`, `RPDE`, and `MDVP:Fo(Hz)`.
- During live inference, audio duration is strictly validated using `librosa.get_duration` (requiring ≥3.0 seconds) to ensure algorithmic stability.

#### 2.2.2. Model Training (`train_voice.py`)

- **Model Architecture:** An `xgboost.XGBClassifier` was selected for its high performance on tabular data.
- **Training Pipeline:** Data was split into an 80/20 stratified set, scaled via `StandardScaler`, and the decision boundary was manually optimized to 0.39 to heavily prioritize clinical sensitivity.
- **Evaluation:** Using robust 5-Fold Cross-Validation, the final tuned model achieved an accuracy of **86.95%** and a massive **98.92% Clinical Sensitivity** (Recall).

### 2.3. Spiral Drawing Model (MobileNetV2)

#### 2.3.1. Image Preprocessing (`crop_to_spiral`)

A strict, multi-step preprocessing pipeline was designed and enforced across both training (`train_spiral.py`) and inference (`app.py`) to eliminate training-serving skew.
- **Process:** The image is converted to grayscale, and a 5x5 Gaussian Blur and inverse Otsu's Binarization are applied to separate ink from background. All ink contours are located using `cv2.findContours`. An absolute bounding box is calculated to tightly crop around the drawing, and a strict 10-pixel padding is added to standardize the input before resizing to 224x224 and applying `mobilenet_v2.preprocess_input`.

#### 2.3.2. Model Training (`train_spiral.py`)

- **Model Architecture:** A `MobileNetV2` model pre-trained on ImageNet with a custom dense classification head.
- **Two-Phase Training Strategy:**
    1.  **Phase 1 (Warm-up):** The entire MobileNetV2 base was frozen; only the custom head was trained for 15 epochs.
    2.  **Fine-Tuning:** The top 50 layers of the MobileNetV2 base were unfrozen. The model was re-compiled with a low learning rate (`1e-5`) and trained for up to 30 additional epochs.
- **Kinematic-Safe Data Augmentation:** Data augmentation was mathematically constrained to strict rotations (≤10°) and flipping with a `nearest` fill mode to preserve true micro-tremors.
- **Evaluation:** The final model evaluated on an unseen test set achieved **82.31% Accuracy** and **81.91% Sensitivity (Recall)**.

---

## 3. System Implementation (`app.py`)

The trained models are deployed within a Flask web application that provides a robust user interface and clinical backend.

### Key Features

- **Clinical Profiling:** Tracks patient age, gender, and session-specific medication states ("ON", "OFF", "UNMEDICATED") to provide contextualized reporting. A dedicated clinician dashboard allows medical professionals to view triage-sorted patient records.
- **Explainable AI (XAI):** Uses Gradient-weighted Class Activation Mapping (Grad-CAM) to generate kinematic heatmaps of the drawings. It superimposes the heatmap on the original drawing and extracts a localized Kinematic Variance score. For voice, Mel-Spectrograms provide visual insights into the acoustic spread.
- **Formal Reporting:** Generates downloadable PDF reports utilizing `fpdf`, consolidating patient data, medication state, voice metrics, and XAI visualizations into clinical documents.
- **Longitudinal Tracking:** Dynamically calculates a per-user session index and powers the `history()` route, which uses Chart.js to visualize the patient's neurological risk trend over multiple sessions.

---

## 4. Conclusion

The NeuroScreen project successfully demonstrates the development and integration of a multi-modal machine learning system for preliminary neurological screening. By combining a high-sensitivity XGBoost model for vocal biomarker analysis with a robust, fine-tuned MobileNetV2 model for motor skills assessment, the platform provides a comprehensive and data-driven risk score. Crucial engineering innovations, including aggressive bounding box isolation, kinematic-safe augmentation, patient demographic tracking, and embedded XAI visualizations, solidify NeuroScreen as a powerful, clinically-focused proof of concept for modern AI health screening tools.
