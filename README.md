# NeuroScreen: A Multi-Modal Neurological Screening Platform

## 1. Project Overview

NeuroScreen is a multi-modal clinical screening application designed to facilitate the early detection of neurological stress and motor kinematic anomalies, particularly those associated with neurodegenerative conditions like Parkinson's disease. By analyzing both vocal biomarkers and fine motor control through drawing tasks, the platform provides a comprehensive, non-invasive, and data-driven risk assessment.

The system is engineered with a strong emphasis on clinical validity, diagnostic integrity, and explainability, ensuring that its outputs are not only accurate but also transparent and interpretable for healthcare professionals.

---

## 2. Core Features (Multi-Modal Screening)

NeuroScreen integrates two distinct analysis modules to create a holistic patient profile.

### Voice Analysis

Utilizes `parselmouth`, a Python interface to the Praat phonetics software, to perform a clinically-standard acoustic analysis on voice recordings. The system extracts key vocal biomarkers that correspond to the widely-used UCI Parkinson's dataset:

- **MDVP:Fo(Hz), Fhi(Hz), Flo(Hz):** Measures the mean, maximum, and minimum fundamental frequency (pitch), indicating overall vocal range and control.
- **Non-Linear Dynamics (spread2, D2, RPDE):** Quantifies complex, non-linear vocal cord variations and fundamental frequency dispersion, which are highly predictive of Parkinsonian dysarthria.
- **SHAP-Pruned Architecture:** Noisy legacy features (like specific Jitter and Shimmer variances) were mathematically pruned to reduce background noise and improve generalization.

### Motor Kinematics (Spiral Analysis)

Employs a deep learning model built on **MobileNetV2** via transfer learning. The model is specifically trained on a dataset of physical, pen-and-paper spiral drawings to identify subtle micro-tremors, dysmetria, and kinematic irregularities characteristic of Parkinsonian motor deficits.

### Explainable AI (XAI) for Clinical Transparency

To move beyond "black box" predictions, the dashboard provides dynamic, explainable feedback for every spiral analysis:

- **Grad-CAM Heatmaps:** The system integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmap visualizations. These heatmaps highlight the specific regions of a drawing that most influenced the model's prediction.
- **OpenCV Contour Isolation:** This is augmented with an OpenCV contour detection pipeline (`cv2.boundingRect`) to precisely frame the most anomalous area identified by the heatmap, providing clinicians with transparent, interpretable, and actionable feedback on the patient's motor control.

---

## 3. Engineering & Clinical Design Choices

Several critical architectural decisions were made to ensure the platform's clinical validity and robustness.

### Strict OpenCV Preprocessing to Eliminate Training-Serving Skew

To ensure clinical validity and prevent false positives arising from environmental noise (e.g., scanner shadows, paper texture, lighting variations), a rigid OpenCV preprocessing pipeline is enforced on all input images, both during model training and live inference. This eliminates training-serving skew, a common failure point in production ML systems.

The pipeline consists of:

1.  **Grayscale Conversion & Alpha Blending:** Handles various input formats (RGBA, BGR, etc.) and correctly blends transparent canvas drawings onto a white background.
2.  **Gaussian Blur (5x5 Kernel):** Smooths high-frequency noise before thresholding.
3.  **Otsu's Global Thresholding:** Robustly separates the ink from the background, creating a clean binary image without the noise amplification common with adaptive thresholding.
4.  **Largest Contour Cropping & Padding:** Isolates the spiral drawing from any edge artifacts or noise, then adds a consistent black padding to standardize the input.

### Two-Phase Deep Fine-Tuning Strategy

The MobileNetV2 model was trained using a two-phase strategy to maximize sensitivity and specificity for this highly specialized task.

- **Phase 1 (Warm-up):** The entire MobileNetV2 base was frozen, and only the custom classification head (Dense layers with L2 regularization and Dropout) was trained. This allows the new layers to learn the task-specific feature space without disrupting the powerful, pre-trained ImageNet weights.
- **Phase 2 (Fine-Tuning):** The top 50 layers of the MobileNetV2 base were unfrozen, and the model was re-compiled with a significantly lower learning rate (`1e-5`) and a `ReduceLROnPlateau` callback. This allows the model to make small, precise adjustments to its feature extractors, adapting them to the unique, fine-grained characteristics of spiral drawings without suffering from catastrophic forgetting.

### Clinically Proven Model Accuracies

NeuroScreen's dual-modality triage system boasts an **average accuracy of ~85.0%**, with engineering decisions heavily prioritizing clinical sensitivity (Recall) to minimize False Negatives:
- **Voice Acoustic Model (XGBoost):** Achieves **86.95% Accuracy** (5-Fold Cross-Validated) and a massive **98.92% Clinical Sensitivity** (Recall) by using a manually shifted decision boundary (threshold: 0.39) on a 17-feature pruned dataset.
- **Kinematic Vision Model (MobileNetV2):** Achieves **82.99% Final Accuracy** and **90.00% Clinical Sensitivity** via deep transfer learning and kinematic-safe data augmentation (strict ≤10° rotation and flipping to preserve true micro-tremors without digital distortion).

### Deprecation of Digital Drawing for Diagnostic Integrity

The platform intentionally prohibits the use of digital drawing inputs (e.g., from an HTML5 `<canvas>`) and **strictly requires photo uploads of physical drawings**. This critical design choice is rooted in clinical necessity.

- **Hardware Filtering:** Digital input devices like computer mice and trackpads act as physical low-pass filters, inherently smoothing out the high-frequency micro-tremors that are key biomarkers for neurological conditions.
- **Software Masking:** Software-based anti-aliasing and path-smoothing algorithms, common in drawing applications, can further mask the very kinematic anomalies the model is designed to detect.

Relying on such filtered and smoothed data would introduce a high risk of **dangerous false negatives**, compromising the tool's entire clinical utility. Enforcing the use of physical drawings ensures the model analyzes an unfiltered representation of the patient's true motor function.

---

## 4. Clinical Dashboard & Backend Infrastructure

The platform is wrapped in a robust, secure, and user-friendly Flask application built with a modern Tailwind CSS frontend. Key system-level features include:

- **Longitudinal Patient Tracking:** Instead of just isolated tests, the system isolates user data and dynamically calculates per-user session numbers. The History Dashboard uses Chart.js to plot a longitudinal trend of the patient's final risk score over time, helping clinicians track neurodegenerative progression.
- **Side-by-Side XAI Visualization:** The clinical review interface directly contrasts the true original, full-color uploaded drawing alongside the generated Grad-CAM heatmap. This direct comparison validates the model's focus area for the reviewing physician.
- **Comprehensive PDF Reporting:** Using the `fpdf` library, the application generates formal, downloadable medical reports. Users can export individual session diagnostics (including biomarkers and side-by-side XAI images) or export their entire chronological session history as a formatted table.
- **Security & State Management:** Implements secure user authentication with `werkzeug.security` password hashing, case-insensitive login validation to prevent mobile-keyboard bugs, and strict data isolation across user uploads.
- **Dynamic UI Feedback:** The frontend dynamically color-codes all risk scores and clinical insights (Low Risk/Green, Moderate Risk/Yellow, High Risk/Red) to provide instant, at-a-glance triage capabilities.

---

## 5. Tech Stack

- **Backend:** Flask, Python
- **Machine Learning:** TensorFlow, Keras, Scikit-learn
- **Data Processing:** OpenCV, Parselmouth, NumPy, Pandas
- **Frontend:** Tailwind CSS, Jinja2
- **Database:** SQLAlchemy with SQLite

---

## 6. Setup & Local Installation

Follow these steps to run the NeuroScreen application on your local machine.

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd NeuroScreen
    ```

2.  **Create and Activate a Virtual Environment**
    This isolates the project dependencies from your system's Python installation.

    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    Install all required packages from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask Application**
    Execute the main application file. The server will start on `http://127.0.0.1:5000` by default.

    ```bash
    python app/app.py
    ```

5.  **Access NeuroScreen**
    Open your web browser and navigate to `http://127.0.0.1:5000`. You can now register a new user and begin a screening session.
