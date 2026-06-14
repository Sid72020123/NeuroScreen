# NeuroScreen: A Multi-Modal Neurological Screening Platform

**Project:** ASEP-2 (Computer Engineering)  
**Team:** 4-Person Student Engineering Team

## 1. Project Overview

NeuroScreen is a multi-modal clinical screening application designed to facilitate the early detection of neurological stress and motor kinematic anomalies, particularly those associated with neurodegenerative conditions like Parkinson's disease. By analyzing both vocal biomarkers and fine motor control through drawing tasks, the platform provides a comprehensive, non-invasive, and data-driven risk assessment.

The system is engineered with a strong emphasis on clinical validity, diagnostic integrity, and explainability, ensuring that its outputs are not only accurate but also transparent and interpretable for healthcare professionals.

---

## 2. Core Features (Multi-Modal Screening)

NeuroScreen integrates two distinct analysis modules to create a holistic patient profile.

### Voice Analysis

Utilizes the `librosa` library to perform acoustic analysis on voice recordings. The system extracts key vocal biomarkers known to be affected by neurological conditions:

- **Jitter (Mean & Local):** Measures the frequency variation from cycle to cycle in vocal fold vibration, indicating phonatory instability.
- **Shimmer (Mean & Local):** Measures the amplitude variation of the sound wave, indicating irregularities in vocal fold vibration.
- **Pitch Standard Deviation:** Quantifies the variability in the fundamental frequency (F0), reflecting overall pitch control.

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
- **Phase 2 (Fine-Tuning):** The top 20 layers of the MobileNetV2 base were unfrozen, and the model was re-compiled with a significantly lower learning rate (`1e-5`) and a `ReduceLROnPlateau` callback. This allows the model to make small, precise adjustments to its feature extractors, adapting them to the unique, fine-grained characteristics of spiral drawings without suffering from catastrophic forgetting.

### Deprecation of Digital Drawing for Diagnostic Integrity

The platform intentionally prohibits the use of digital drawing inputs (e.g., from an HTML5 `<canvas>`) and **strictly requires photo uploads of physical drawings**. This critical design choice is rooted in clinical necessity.

- **Hardware Filtering:** Digital input devices like computer mice and trackpads act as physical low-pass filters, inherently smoothing out the high-frequency micro-tremors that are key biomarkers for neurological conditions.
- **Software Masking:** Software-based anti-aliasing and path-smoothing algorithms, common in drawing applications, can further mask the very kinematic anomalies the model is designed to detect.

Relying on such filtered and smoothed data would introduce a high risk of **dangerous false negatives**, compromising the tool's entire clinical utility. Enforcing the use of physical drawings ensures the model analyzes an unfiltered representation of the patient's true motor function.

---

## 4. Tech Stack

- **Backend:** Flask, Python
- **Machine Learning:** TensorFlow, Keras, Scikit-learn
- **Data Processing:** OpenCV, Librosa, NumPy, Pandas
- **Frontend:** Tailwind CSS, Jinja2
- **Database:** SQLAlchemy with SQLite

---

## 5. Setup & Local Installation

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
