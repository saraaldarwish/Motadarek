MOTADAREK: Audio-Radar Fall Detection and Severity Assessment System
Live Demo

🚀 HuggingFace Deployment:
https://huggingface.co/spaces/SarahSulaiman/Motadarek

Overview

MOTADAREK is a multimodal AI-powered fall detection and severity assessment system that combines audio analysis, radar-based recovery detection, and contextual classification models to analyze fall events and assess potential severity.

The system integrates multiple machine learning models into a unified inference pipeline using FastAPI and HuggingFace Spaces deployment.

Features
Fall audio detection
Human position classification
Surface type classification
Scream detection
Radar recovery analysis
Rule-based fusion logic
Interactive 3D visualization interface
Real-time inference pipeline
System Pipeline
User uploads audio and/or radar inputs.
Fall detector analyzes the primary audio.
If a fall is detected:
Position model runs
Surface classifier runs
Scream detector runs
Radar recovery model runs
Rule-based fusion combines outputs.
Final assessment and recovery status are generated.
Technologies Used
Backend
FastAPI
Python
Uvicorn
Frontend
HTML
CSS
JavaScript
Machine Learning
PyTorch
HuggingFace Transformers
AST (Audio Spectrogram Transformer)
Scikit-learn
Librosa
Joblib
Deployment
HuggingFace Spaces
Docker
Project Structure
.
├── static/
│   └── frontend files
├── notebooks/
│   └── model training notebooks
├── model/
│   ├── fall detector
│   ├── position classifier
│   ├── surface classifier
│   ├── scream detector
│   └── radar recovery model
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
Deployment and Repository

GitHub contains the project source code, documentation, and version history.
HuggingFace Spaces hosts the deployed system, allowing users to run the application and test the models through a live web interface.

Notes

Large model weight files are stored in the deployed HuggingFace environment rather than the GitHub repository to keep the repository lightweight and manageable.

