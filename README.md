# Sign Language Detector using Digital Image Processing

A real-time Indian Sign Language (ISL) recognition system built using Python, OpenCV, MediaPipe, and Machine Learning. The project detects hand landmarks from a webcam, extracts meaningful features, and predicts the corresponding ISL alphabet in real time.

---

## Project Overview

This project is a Digital Image Processing (DIP) application that recognizes Indian Sign Language (ISL) gestures in real time using image preprocessing, hand landmark extraction, and machine learning. The system captures hand gestures through a webcam, applies multiple image enhancement techniques, extracts hand landmarks, and predicts the corresponding ISL alphabet.

The project demonstrates the application of Digital Image Processing techniques to improve gesture recognition accuracy under different lighting and environmental conditions.

---

## Features

- Real-time ISL alphabet recognition
- Hand landmark detection using MediaPipe
- Random Forest based gesture classification
- Image preprocessing for improved robustness
- Reference image displayed alongside webcam feed
- Complete training pipeline included
- Privacy-friendly repository (no personal images)
  
---

## Technologies Used

- Python
- Digital Image Processing (DIP)
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Pickle

---

## Project Structure

```text
sign-language-detector/
├── collect_images.py
├── create_dataset.py
├── train_classifier.py
├── inference_classifier.py
├── data.pickle
├── model.p
├── ref.jpg
├── requirements.txt
└── README.md
```

---

## Workflow

1. Collect gesture images using `collect_images.py`
2. Generate landmark dataset using `create_dataset.py`
3. Train the classifier using `train_classifier.py`
4. Run `inference_classifier.py` for real-time prediction

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train_classifier.py
```

This generates:

```
model.p
```

### Start real-time recognition

```bash
python inference_classifier.py
```

The webcam opens and displays:

- Live hand detection
- Predicted ISL alphabet
- Reference ISL image for comparison

To exit the application, press **Ctrl + C** in the terminal.

---

## Image Preprocessing

The project improves recognition accuracy using several preprocessing techniques:

- Histogram Equalization
- CLAHE
- Gaussian Blur
- Median Blur
- Bilateral Filtering
- Brightness Adjustment
- Contrast Enhancement
- Gamma Correction
- Sharpening Filter
- Edge Enhancement

---

## Digital Image Processing Pipeline

- Image Collection
- Hand Landmark Extraction
- Dataset Creation
- Random Forest Training
- Real-Time Prediction

---
