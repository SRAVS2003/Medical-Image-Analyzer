# Medical Image Analyzer

Full-stack **medical imaging** project for AI/DS portfolios: **classical OpenCV** + **deep learning (ResNet18)** with **Grad-CAM**, served through **FastAPI** and a web UI.

**Educational / research demo — not for clinical diagnosis.**

## Features

| Tab | Module | Description |
|-----|--------|-------------|
| **A — Live camera** | OpenCV | Webcam stream: original / enhanced / region boxes |
| **B — OpenCV upload** | Classical CV | CLAHE enhancement, Otsu segmentation, bounding boxes + report |
| **C — AI + Grad-CAM** | Deep learning | Chest X-ray-style **NORMAL vs PNEUMONIA** demo classifier + **explainability heatmap** |

## Tech stack

Python, OpenCV, PyTorch, FastAPI, HTML/CSS/JS

## Setup (Windows)

```powershell
cd C:\Users\moorkond\Desktop\projects\medical-image-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### One-time: create AI model weights

Tab **C** needs a weights file. Quick demo (uses bundled sample images):

```powershell
python ml/train_demo.py
```

For **real metrics**, download the [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset into `data/chest_xray/` and run:

```powershell
python ml/train_chest_xray.py --data-dir data/chest_xray --epochs 5
```

## Run the app

```powershell
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** (use **8001** if 8000 is busy).

Or double-click **`run.bat`**.

## Project layout

```
medical-image-analyzer/
  backend/           # FastAPI server, OpenCV routes, AI routes
  frontend/          # Web UI (HTML/CSS/JS)
  ml/                # PyTorch model, training, Grad-CAM inference
  models/            # Saved weights (*.pth) — not committed if large
  sample-images/     # Demo PNGs for OpenCV + quick train
  data/              # Your Kaggle dataset (gitignore)
```

## AI/DS story (for reports & GitHub)

1. **Baseline:** OpenCV preprocessing + classical segmentation  
2. **Model:** Transfer learning (ResNet18), binary classification  
3. **Explainability:** Grad-CAM visualization in the UI  
4. **Evaluation:** `train_chest_xray.py` prints classification report + ROC-AUC on **test** split  

## Disclaimer

This project is for **learning and portfolio demonstration** only. It must not be used for real medical decisions without proper validation, regulatory review, and clinical workflows.

## Author

[SRAVS2003](https://github.com/SRAVS2003)
