# Industrial QC Inspector ⚙️

An automated Quality Control (QC) system designed to detect defects in mechanical parts (such as pump impellers) using Deep Learning. This application uses a Convolutional Neural Network (CNN) to analyze surface integrity and classify parts as either "OK" or "Defective" in real-time.

## 🚀 Overview
In industrial manufacturing, manual inspection is often slow and prone to human error. This project provides a Streamlit-based interface that allows users to upload high-resolution images of casted parts. The underlying model analyzes the pixels for structural anomalies, cracks, or deformations, providing an immediate pass/fail result with a confidence metric.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Deep Learning Framework:** TensorFlow / Keras
* **Model Architecture:** MobileNetV2 (Transfer Learning)
* **Image Processing:** PIL (Pillow), NumPy

## 📂 Project Structure
* `app.py`: The main Streamlit application script containing the UI and preprocessing logic.
* `industrial_defect_model.keras`: The trained Deep Learning model file.
* `requirements.txt`: List of Python dependencies (TensorFlow, Streamlit, Pillow, NumPy).

## 🔧 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd industrial-qc-inspector
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 🖥️ Usage
1.  Launch the app in your browser (usually at `http://localhost:8501`).
2.  Upload a `.jpg`, `.jpeg`, or `.png` image of the mechanical part.
3.  The system will automatically:
    * Convert the image to RGB.
    * Resize it to 224x224 pixels.
    * Normalize pixel values for the model.
    * Display the classification result (PASS/FAIL) and confidence score.

## 🧠 Model Details
The model is optimized for binary classification:
* **Class 0:** Defective (Structural anomalies detected).
* **Class 1:** OK (Surface integrity confirmed).
* Preprocessing includes image normalization to a [0, 1] range to ensure consistent inference performance.

---
*Developed as part of an Industrial AI initiative for automated manufacturing inspection.*
