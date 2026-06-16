# NeuroScreen: Project Technology Explained

This document explains the technology behind the NeuroScreen project in simple English.

## What is NeuroScreen? (Project Overview)

NeuroScreen is a web application that helps screen for early signs of neurological stress. It works like a preliminary check-up tool. It analyzes two things: a person's voice and a drawing they make. By looking at both, it can spot tiny issues that might be linked to conditions like Parkinson's disease, providing a data-driven risk score.

The main goal is to provide an easy-to-use, non-invasive tool that gives early warnings, helping people know if they should seek a full clinical evaluation.

---

## How It Works (Core Features)

NeuroScreen uses two different methods to check for neurological signs.

### 1. Voice Analysis

The app listens to a recording of your voice. It's not listening to _what_ you say, but _how_ you say it. It measures:

- **Jitter:** Tiny, rapid variations in your voice's pitch (how high or low it is).
- **Shimmer:** Tiny, rapid variations in your voice's volume (how loud or soft it is).
- **Harmonics-to-Noise Ratio (HNR):** The ratio of "clean" sound to "noisy" sound in your voice, which can relate to hoarseness.

Instability in these areas can be an early indicator of neurological stress.

### 2. Spiral Drawing Analysis

The app uses an Artificial Intelligence (AI) model to analyze a photo of a spiral that you draw on a piece of paper. This AI is trained to look for:

- **Micro-tremors:** Very small, shaky movements in the drawing that are hard to see with the naked eye.
- **Kinematic Anomalies:** General irregularities in the drawing's shape, smoothness, and consistency.

### 3. Explainable AI (XAI)

We don't want the AI to be a "black box." To make its decisions clear, the app shows you _why_ it flagged a drawing as risky. It generates a special image that:

- **Creates a Heatmap:** It overlays colors on your drawing, with "hot" colors (like red) showing the exact parts that the AI found most unusual.
- **Draws a Box:** It uses computer vision to draw a rectangle around the most significant area of concern, pointing it out directly.

---

## Important Design Choices

We made three very important technical decisions to make sure the app is reliable and clinically valid.

### 1. Strict Image Cleanup

Before the AI analyzes a drawing, the image goes through a strict, automatic cleanup process. It's converted to black and white, blurred slightly to remove noise, and the drawing is perfectly isolated from the background. This is crucial because it prevents things like scanner shadows or the texture of the paper from being mistaken for a tremor, which would lead to a false positive. Every image is treated the same way, ensuring fair analysis.

### 2. Smart, Two-Step AI Training

The AI model was trained in two phases to make it highly accurate.

- **Phase 1 (Warm-up):** The AI first learned general image recognition.
- **Phase 2 (Fine-Tuning):** We then "fine-tuned" the AI by training it specifically on thousands of spiral drawings. We used a very slow and careful training process, which made the model an expert at spotting the unique, subtle patterns found in drawings by people with motor control issues.

### 3. No Digital Drawings Allowed (Only Real Paper)

The app _only_ accepts photos of spirals drawn on physical paper. We do not allow drawings made with a mouse or a digital tablet. This is a critical safety feature.

- **Why?** A computer mouse or a digital pen automatically smooths out your hand movements. This smoothing effect hides the exact micro-tremors and shakiness that the app is designed to detect. Using a digital drawing would create a high risk of a **false negative** (telling someone they are fine when a problem might exist), which is dangerous. Using real pen and paper provides a true, unfiltered sample of a person's motor skills.

---

## Technology Stack

- **Backend:** Flask, Python
- **Machine Learning:** TensorFlow, Keras, Scikit-learn
- **Data Processing:** OpenCV, Parselmouth, NumPy, Pandas
- **Frontend:** Tailwind CSS, Jinja2
- **Database:** SQLAlchemy with SQLite

---

## Detailed Model & Dataset Information

This section provides a deeper technical dive into the machine learning models and the datasets used to train them.

### Spiral Drawing Model (MobileNetV2 Transfer Learning)

- **Model Architecture:** A **MobileNetV2** model, pre-trained on ImageNet, with a custom classification head. The top 20 layers of the base model were fine-tuned for this specific task.
- **Training Strategy:** A two-phase training process was used to maximize accuracy and prevent catastrophic forgetting.
    1.  **Warm-up Phase:** The entire MobileNetV2 base was frozen, and only the custom head (Dense layers with L2 regularization and Dropout) was trained. This allowed the new layers to adapt to the feature space of spiral drawings.
    2.  **Fine-tuning Phase:** The top 20 layers of the MobileNetV2 base were unfrozen. The model was then re-compiled with a very low learning rate (`1e-5`) and trained with a `ReduceLROnPlateau` callback for precise, stable adjustments.

- **Combined Training Dataset:** The model was trained on a composite dataset aggregated from multiple public sources to ensure diversity. The total dataset consists of approximately **600 real images** before any data augmentation is applied.
    - **Original HandPD Dataset:**
        - **Contribution:** 368 spiral images.
        - **Distribution:** 72 Healthy + 296 Patient.

    - **NewHandPD Dataset:**
        - **Contribution:** ~130 spiral images.
        - **Distribution:** Sourced from roughly 35 Healthy and 31 Patient subjects, with multiple spirals each.

    - **Kaggle "Handwritten Spirals" (kmader):**
        - **Contribution:** 102 spiral images.
        - **Distribution:** 51 Healthy + 51 Patient.
        - **Integration Method:** All images from the `spiral/training/healthy/`, `spiral/testing/healthy/`, `spiral/training/parkinson/`, and `spiral/testing/parkinson/` folders were copied directly into the project's combined `healthy/` and `parkinsons/` training directories.

### Voice Analysis Model (XGBoost)

- **Model Architecture:** An **XGBoost Classifier**, a powerful and accurate machine learning model. It was automatically tuned to find the best settings for this specific task.
- **Features Used for Training:** The model was trained on 7 core acoustic features extracted using `parselmouth` (a tool that uses the clinical-standard Praat software). This ensures our measurements are consistent with established medical research. The features include:
    1.  Mean, Max, and Min Pitch
    2.  Jitter (Pitch variation)
    3.  Shimmer (Volume variation)
    4.  Noise-to-Harmonics Ratio (NHR)
    5.  Harmonics-to-Noise Ratio (HNR)
- **Dataset:** Trained on a combination of the public **UCI Parkinson's Dataset** and another public audio collection. All data was processed identically to ensure consistency. A `StandardScaler` is used to normalize the features before they are fed to the model.
