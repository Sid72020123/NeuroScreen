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

- **spread2:** Measures how spread out or varied your vocal pitches are.
- **D2 (Correlation Dimension):** Measures the complexity or "roughness" in the voice signal.
- **RPDE (Recurrence Period Density Entropy):** A mathematical way to measure how well the vocal cords are maintaining a steady, repeating rhythm.

Instability in these non-linear areas can be an early indicator of neurological stress.

### 2. Spiral Drawing Analysis

The app uses an Artificial Intelligence (AI) model to analyze a photo of a spiral that you draw on a piece of paper. This AI is trained to look for:

- **Micro-tremors:** Very small, shaky movements in the drawing that are hard to see with the naked eye.
- **Kinematic Anomalies:** General irregularities in the drawing's shape, smoothness, and consistency.

### 3. Explainable AI (XAI)

We don't want the AI to be a "black box." To make its decisions clear, the app shows you _why_ it flagged a drawing as risky. It generates a special image that:

- **Creates a Heatmap:** It overlays colors on your drawing, with "hot" colors (like red) showing the exact parts that the AI found most unusual.
- **Draws a Box:** It uses computer vision to draw a rectangle around the most significant area of concern, pointing it out directly.
- **Voice Spectrograms:** It generates a visual representation of your audio recording, showing the frequencies of your voice over time.

---

## Features Designed for You and Your Doctor

- **Track Your Progress Over Time:** NeuroScreen doesn't just look at a single snapshot. It keeps a history of all your past tests and draws a trend graph. It also allows you to track whether you were taking medication ("ON", "OFF", or "UNMEDICATED") during a test, which provides context.
- **Downloadable Doctor Reports:** With a single click, you can generate a professional, formatted PDF report of your test results. This PDF includes all your voice measurements, demographics, medication state, risk scores, and the side-by-side images of your spiral drawing and the AI heatmap.
- **Clinician Portal:** Healthcare professionals can log in with special credentials to monitor all their patients from a unified dashboard, sorted by the patients who need the most immediate attention.
- **Side-by-Side Image Review:** When looking at your results, the app displays your original, normal drawing right next to the AI's "Heatmap" version.

---

## Important Design Choices

### 1. Strict Image Cleanup

Before the AI analyzes a drawing, the image goes through a strict, automatic cleanup process. It's converted to black and white, blurred slightly to remove noise, and the drawing is perfectly isolated from the background by tightly cropping around the ink strokes. Every image is treated the exact same way with a strict 10-pixel border, ensuring fair analysis and preventing false positives from paper texture or shadows.

### 2. Smart, Two-Step AI Training

The AI model was trained in two phases to make it highly accurate.
- **Phase 1 (Warm-up):** The AI first learned general image recognition.
- **Phase 2 (Fine-Tuning):** We then "fine-tuned" the AI by training it specifically on thousands of spiral drawings. We used a very slow and careful training process, making the model an expert at spotting the unique, subtle patterns found in drawings by people with motor control issues.

### 3. No Digital Drawings Allowed (Only Real Paper)

The app _only_ accepts photos of spirals drawn on physical paper. We do not allow drawings made with a mouse or a digital tablet to avoid digital smoothing filters from masking true micro-tremors.

---

## Detailed Model & Dataset Information

### Spiral Drawing Model (MobileNetV2 Transfer Learning)

- **Model Architecture:** A **MobileNetV2** model, pre-trained on ImageNet, with a custom classification head. The top 50 layers of the base model were fine-tuned for this specific task.
- **Kinematic-Safe Data Augmentation:** Because deep learning needs a lot of data, we mathematically multiplied our dataset by rotating (up to 10 degrees) and flipping the images. We intentionally avoided "stretching" or "shearing" the images so the AI learned to spot true human micro-tremors instead of digital distortion.
- **Accuracy:** The final model achieved **82.31% Accuracy** and **81.91% Sensitivity**.
- **Combined Training Dataset:** Trained on a composite dataset aggregated from HandPD, NewHandPD, and Kaggle (kmader), totaling approximately 734 images.

### Voice Analysis Model (XGBoost)

- **Model Architecture:** An **XGBoost Classifier**.
- **Features Used for Training:** The model was originally tested with 22 features, but we used a technique called SHAP to mathematically prove that some features were just adding "noise." We pruned the model down to **17 core acoustic features**, which made it much more accurate.
- **Accuracy:** By manually adjusting the decision threshold down to 0.39, we ensured the model catches almost everyone who is sick, resulting in a **98.92% Clinical Sensitivity** (Recall) and **86.95% Overall Accuracy**.
