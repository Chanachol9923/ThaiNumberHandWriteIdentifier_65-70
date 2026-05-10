# ThaiNumberHandWriteIdentifier

A high-performance Thai handwriting recognition system focusing on numbers **๖๕–๗๐**. Built with **Machine Learning (SVM)** and **FastAPI**, featuring a dedicated management suite and a sleek **Bingus-themed** user interface.

## ✨ Key Features
* **Thai Handwriting Recognition**: Precise prediction of Thai numerals: ๖๕, ๖๖, ๖๗, ๖๘, ๖๙, and ๗๐.
* **Real-time Training Console**: Integrated 400px terminal log streaming directly to the browser via Server-Sent Events (SSE).
* **Admin Hot-swap**: Switch between different trained models (`.pkl`) or upload new ones instantly without a server restart.
* **Dataset Management**: Draw, categorize, and save handwritten samples directly into the local dataset.
* **Dataset Export**: One-click ZIP compression to download the entire collected dataset.
* **Smart Preprocessing**: Automatic image handling including inversion, auto-cropping, and center-of-mass alignment.

## 🛠️ Technical Stack
* **Backend**: Python, FastAPI, Uvicorn
* **Machine Learning**: Scikit-learn (SVM - RBF Kernel), Joblib, NumPy
* **Image Processing**: Pillow (PIL), SciPy (ndimage)
* **Frontend**: HTML5 Canvas, Vanilla JavaScript (Modern UI with CSS Variables)

## 🚀 Getting Started

1. Clone the repository
`bash
git clone [https://github.com/Chanachol9923/ThaiNumberHandWriteIdentifier.git](https://github.com/Chanachol9923/ThaiNumberHandWriteIdentifier.git)
cd ThaiNumberHandWriteIdentifier`

2. Install dependencies
`pip install -r requirements.txt`

3. Have fun on your http://127.0.0.1:8000!
`uvicorn main:app --reload`


📂 Project Structure
main.py: Core FastAPI server handling API endpoints and real-time streaming.

BingusSchoolForTraining.py: Training script including feature extraction logic (Zone features, Projections, Diagonals).

dataset/: Directory for categorized image samples (labeled 65-70).

models/: Directory for stored model files (.pkl).

static/: Web UI assets including index.html and project images.

