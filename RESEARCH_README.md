# NeuroScreen: A Multi-Modal AI-Powered Platform for Neurological Stress Screening

## 1. Abstract

This paper details the architecture and implementation of NeuroScreen, a multi-modal, non-invasive screening tool designed for the early detection of neurological stress indicators, particularly those associated with neurodegenerative conditions like Parkinson's disease. The system integrates two distinct machine learning models to analyze vocal biomarkers and fine motor control, providing a holistic, data-driven risk assessment. The voice analysis module employs an XGBoost classifier trained on clinically relevant acoustic features extracted via the Praat engine, while the motor skills module uses a fine-tuned MobileNetV2 deep learning model to analyze spiral drawings. The platform is delivered as a Flask web application, featuring an explainable AI (XAI) component to ensure transparency and interpretability of the results. The primary objective of this work is to demonstrate a robust, accessible, and clinically-grounded proof-of-concept for preliminary neurological screening.

---

## 2. Methodology

The project's methodology is bifurcated into two primary analytical streams: voice analysis and motor kinematics (spiral) analysis. Each stream utilizes distinct datasets, feature extraction techniques, and model architectures.

### 2.1. Datasets

A composite data strategy was employed to train and validate the models, ensuring a diverse and robust training corpus.

#### 2.1.1. Voice Analysis Datasets

- **UCI Parkinson's Dataset:** This foundational dataset consists of 195 samples with 22 pre-extracted acoustic features per voice recording. It served as the primary source for training the voice classification model.
- **Figshare 2023 (Prior et al.):** This dataset contains raw `.wav` audio files from both healthy and Parkinson's-diagnosed individuals. These files were used to augment the training data and, critically, to validate that our feature extraction pipeline (`extract_praat_features`) could replicate the feature space of the UCI dataset from raw audio.

#### 2.1.2. Spiral Drawing Datasets

A combined dataset of **734 images** was aggregated from multiple public sources to train the spiral analysis model. This approach mitigates dataset-specific biases and improves model generalization.

- **HandPD & NewHandPD (UNESP Brazil):** Provided the core set of spiral and wave drawings from healthy and patient cohorts.
- **Kaggle Parkinson's Drawings (kmader):** Contributed an additional 102 spiral images, further diversifying the training set.

### 2.2. Voice Analysis Model (XGBoost Classifier)

#### 2.2.1. Feature Extraction (`extract_praat_features`)

A custom feature extraction function was developed to process raw `.wav` files, ensuring consistency between the Figshare dataset and the UCI dataset's methodology.

- **Tooling:** The function uses the `parselmouth` Python library, a wrapper for the Praat phonetics software, which is a gold standard in acoustic analysis.
- **Process:** For each audio file, the function loads the sound and calculates the following 7 core features in a specific order to match the model's input requirements:
    1.  **MDVP:Fo(Hz):** Mean fundamental frequency (average pitch).
    2.  **MDVP:Fhi(Hz):** Maximum fundamental frequency (highest pitch).
    3.  **MDVP:Flo(Hz):** Minimum fundamental frequency (lowest pitch).
    4.  **MDVP:Jitter(%):** A measure of frequency variation from cycle to cycle.
    5.  **MDVP:Shimmer:** A measure of amplitude variation from cycle to cycle.
    6.  **NHR (Noise-to-Harmonics Ratio):** The ratio of noise to tonal components in the voice.
    7.  **HNR (Harmonics-to-Noise Ratio):** The ratio of tonal components to noise.

#### 2.2.2. Model Training (`train_voice.py`)

- **Model Architecture:** An `xgboost.XGBClassifier` was selected for its high performance on tabular data.
- **Training Pipeline:**
    1.  **Data Merging:** The UCI dataset and the features extracted from the Figshare dataset were concatenated into a single master DataFrame.
    2.  **Data Splitting:** The data was split into an 80% training set and a 20% testing set, stratified to maintain the original class distribution.
    3.  **Scaling:** A `sklearn.preprocessing.StandardScaler` was fitted on the training data to normalize the feature values, and this scaler was subsequently used on the test data.
    4.  **Hyperparameter Tuning:** `sklearn.model_selection.GridSearchCV` was employed with 5-fold cross-validation to systematically search for the optimal hyperparameters (`n_estimators`, `max_depth`, `learning_rate`). The best parameters were found to be `{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}`.
- **Evaluation:** The final tuned model achieved an accuracy of **94.87%** on the unseen test set. The model demonstrated strong performance in identifying both healthy (Recall: 0.97) and Parkinson's (Recall: 0.90) samples, as detailed in its classification report.

### 2.3. Spiral Drawing Model (MobileNetV2)

#### 2.3.1. Image Preprocessing (`preprocess_image`)

A strict, multi-step preprocessing pipeline was designed and enforced across both training (`train_spiral.py`) and inference (`app.py`) to eliminate training-serving skew.

- **Process:**
    1.  The image is loaded and converted to grayscale.
    2.  A 5x5 Gaussian Blur is applied to reduce high-frequency noise.
    3.  Otsu's Binarization (inverse) is used to create a high-contrast binary image, cleanly separating the drawing (white) from the background (black).
    4.  The binary image is converted back to a 3-channel RGB format.
    5.  The image is resized to 224x224 pixels.
    6.  Finally, `mobilenet_v2.preprocess_input` is applied to normalize the pixel values according to the model's requirements.

#### 2.3.2. Model Training (`train_spiral.py`)

- **Model Architecture:** A transfer learning approach was used, building upon a `MobileNetV2` model pre-trained on ImageNet. A custom classification head was added:
    - `GlobalAveragePooling2D()`
    - `Dense(128, activation='relu', kernel_regularizer=l2(0.01))`
    - `Dropout(0.5)`
    - `Dense(1, activation='sigmoid')`
- **Two-Phase Training Strategy:**
    1.  **Phase 1 (Warm-up):** The entire MobileNetV2 base was frozen, and only the custom head was trained for 15 epochs. This allows the new layers to adapt to the feature space of spiral drawings without corrupting the pre-trained weights.
    2.  **Phase 2 (Fine-Tuning):** The top 20 layers of the MobileNetV2 base were unfrozen. The model was re-compiled with a very low learning rate (`1e-5`) and trained for up to 30 additional epochs. This allows the model to make small, precise adjustments to its feature extractors. `EarlyStopping` and `ReduceLROnPlateau` callbacks were used for stable convergence.
- **Class Imbalance:** The significant class imbalance in the dataset (471 Parkinson's vs. 263 Healthy) was addressed by providing `class_weight='balanced'` to the model during training, which increases the penalty for misclassifying the minority class.
- **Evaluation:** The final model was evaluated on an unseen test set of 147 images.
    - **Overall Accuracy:** **79.59%**
    - **Classification Report:** The model achieved a **recall (sensitivity) of 0.87** for the Parkinson's class, which is a critical metric for a screening tool, as it indicates a low rate of false negatives. The precision for this class was 0.82.

---

## 3. System Implementation (`app.py`)

The trained models are deployed within a Flask web application that provides a user interface for screening and result interpretation.

### Key Functions

- **`get_voice_dependencies()` / `get_spiral_model()`**
    - **Purpose:** These functions load the XGBoost model, StandardScaler, and the TensorFlow/Keras model into memory upon application startup.
    - **Mechanism:** They use global variables to cache the models, preventing the high latency of reloading them from disk on every user request.

- **Authentication & Security**
    - **Purpose:** Manages user accounts and secures patient data.
    - **Mechanism:** Implements `flask_login` with `werkzeug.security` password hashing. Custom logic ensures case-insensitive login evaluation to prevent common mobile input errors.

- **`upload_voice()`**
    - **Purpose:** This Flask route handles the voice analysis workflow.
    - **Mechanism:** It receives a `.wav` file, validates its length using `librosa.get_duration` to ensure mathematical stability, and passes it to `extract_praat_features`. The 22-feature array is pruned to the 17 optimal features, scaled, and evaluated by the XGBoost model.

- **`upload_spiral(session_id)`**
    - **Purpose:** This route manages the spiral drawing analysis.
    - **Mechanism:** Executes the exact OpenCV preprocessing pipeline used during training to prevent training-serving skew. It also intercepts and saves the raw, full-color original image to provide a baseline for the clinician to compare against the AI's heatmap.

- **`generate_gradcam(img_array, base_img, model, ...)`**
    - **Purpose:** Implements Explainable AI (XAI) for the spiral model.
    - **Mechanism:** It uses the Gradient-weighted Class Activation Mapping (Grad-CAM) technique. By accessing the gradients and output of the final convolutional layer (`out_relu`), it computes a heatmap that highlights which pixels in the input image were most influential in the model's decision. This heatmap is then color-mapped and superimposed on the original drawing, and a bounding box is drawn around the most salient contour.

- **`generate_clinical_analysis(features_dict)`**
    - **Purpose:** Provides human-readable interpretation of the voice analysis results.
    - **Mechanism:** It compares the measured values for `MDVP:Jitter(%)` and `MDVP:Shimmer` against clinically-informed thresholds stored in the `VOICE_ANALYSIS_THRESHOLDS` dictionary. It then generates simple English sentences indicating whether the values are elevated or within the expected range.

- **`build_report_pdf(session_record)` & `generate_history_report()`**
    - **Purpose:** Creates a formal, downloadable PDF summary of a screening session.
    - **Mechanism:** Programmatically constructs PDF medical records. The individual report includes side-by-side original and XAI images, voice metric tables, and human-readable AI analysis. The history report aggregates a user's entire chronological screening history into a structured triage table.

- **`serialize_session(session_record)` & Longitudinal Tracking**
    - **Purpose:** Prepares database records for frontend rendering and calculates patient progression.
    - **Mechanism:** Dynamically calculates a per-user session index (e.g., Session #1, #2) rather than exposing the global database primary key. This powers the `history()` route, which feeds a Chart.js instance to visualize the patient's longitudinal risk trend over time.

---

## 4. Conclusion

The NeuroScreen project successfully demonstrates the development and integration of a multi-modal machine learning system for preliminary neurological screening. By combining a high-accuracy XGBoost model for vocal biomarker analysis with a robust, fine-tuned MobileNetV2 model for motor skills assessment, the platform provides a comprehensive and data-driven risk score. Key engineering decisions, including the enforcement of a strict preprocessing pipeline to eliminate training-serving skew, a two-phase deep learning strategy to maximize model sensitivity, and the integration of Grad-CAM for explainability, contribute to the system's clinical validity and transparency. NeuroScreen serves as a powerful proof-of-concept for how modern AI techniques can be applied to create accessible, non-invasive, and interpretable health screening tools.
