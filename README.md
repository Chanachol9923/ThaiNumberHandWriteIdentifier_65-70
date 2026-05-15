# Thai Number Handwrite Identifier 65-70

A Thai handwriting recognition system for Thai numerals **๖๕ (65) to ๗๐ (70)**. Built with **SVM (RBF Kernel)** and **FastAPI**, featuring real-time prediction, data collection, model training, and model management via a web UI.

## Features

- **Handwriting Prediction** — Draw Thai numerals on a canvas and get real-time prediction with confidence score
- **Data Collector** — Draw and save labeled samples to expand the training dataset
- **Training Pipeline** — Train an SVM model directly from the browser with live terminal streaming (SSE)
- **Model Management** — Upload `.pkl` models or switch between trained models without server restart
- **Smart Preprocessing** — Auto-invert, crop, center-of-mass alignment, and resize to 28x28
- **Data Augmentation** — Rotation, translation, and noise injection during training
- **Feature Extraction** — Raw pixels, 4x4 zone features (mean + std), horizontal/vertical projections, diagonal traces

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **ML:** scikit-learn (SVC, RBF kernel, probability=true), joblib, NumPy
- **Image Processing:** Pillow, SciPy (ndimage.center_of_mass)
- **Frontend:** HTML5 Canvas, Vanilla JavaScript, CSS custom properties

## Getting Started

```bash
git clone https://github.com/Chanachol9923/ThaiNumberHandWriteIdentifier_65-70.git
cd ThaiNumberHandWriteIdentifier_65-70
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

## Live Demo

Deployed on Vercel: [thai-number-hand-write-identifier.vercel.app](https://thai-number-hand-write-identifier-qmnvuddbw.vercel.app)

## Project Structure

| Path | Description |
|------|-------------|
| `main.py` | FastAPI server — prediction, dataset save, model upload/switch, training streaming |
| `BingusSchoolForTraining.py` | SVM training script with augmentation and feature extraction |
| `dataset/` | Collected handwriting samples organized by label (65–70) |
| `models/` | Trained `.pkl` model files |
| `static/` | Frontend assets (index.html, images) |
| `vercel.json` | Vercel serverless deployment config |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/predict` | Predict a drawn digit (base64 image) |
| POST | `/save-sample` | Save a drawn sample to dataset |
| GET | `/get-count/{label}` | Get sample count for a label |
| GET | `/list-models` | List available models |
| POST | `/select-model` | Switch active model |
| POST | `/update-model` | Upload a new `.pkl` model |
| GET | `/download-model` | Download current model |
| GET | `/download-dataset` | Download all collected samples as ZIP |
| GET | `/train-stream` | Start training with SSE output stream |


