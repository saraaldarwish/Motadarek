# MOTADAREK  
## Audio-Radar Fall Detection and Severity Assessment System

🚀 **Live Demo:**  
https://huggingface.co/spaces/SarahSulaiman/Motadarek

---

## Overview

MOTADAREK is a multimodal AI-powered fall detection and severity assessment system. It combines audio analysis, radar-based recovery detection, and contextual classification models to analyze fall events and assess potential severity.

The system integrates multiple machine learning models into one inference pipeline using FastAPI and HuggingFace Spaces.

## Repository and Deployment

GitHub contains the project code and notebooks.

HuggingFace Spaces contains the live deployed application, including the model files needed for online inference.

---

## Features

- Fall audio detection
- Human position classification
- Surface type classification
- Scream detection
- Radar recovery analysis
- Rule-based severity fusion
- Interactive 3D visualization interface
- Real-time inference pipeline

---

## System Pipeline

1. The user uploads audio and/or radar inputs.
2. The fall detector analyzes the primary fall audio.
3. If a fall is detected, the system runs:
   - Position classification
   - Surface classification
   - Scream detection
   - Radar recovery detection
4. Rule-based fusion combines the model outputs.
5. The final severity assessment and recovery status are generated.

---

## Technologies Used

### Backend
- Python
- FastAPI
- Uvicorn

### Frontend
- HTML
- CSS
- JavaScript

### Machine Learning
- PyTorch
- HuggingFace Transformers
- AST (Audio Spectrogram Transformer)
- Scikit-learn
- Librosa
- Joblib

### Deployment
- HuggingFace Spaces
- Docker

---

## Project Structure

```text
.
├── static/
│   └── frontend files
├── notebooks/
│   └── model training notebooks
├── model/
│   ├── fall_detector/
│   ├── position/
│   ├── surface/
│   ├── scream/
│   └── radar/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
