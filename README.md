# Medical Image Analyzer

Educational full-stack project for **AI / Data Science** portfolios: **OpenCV** (enhancement, segmentation) + **PyTorch** (ResNet18, Grad-CAM) with a **FastAPI** backend and web UI.

## Features

- **Live camera** – OpenCV enhancement / region demo
- **Upload** – classical CV pipeline (enhance + annotate + report)
- **AI tab** – demo classification + explainability heatmap

## Tech stack

Python, OpenCV, PyTorch, FastAPI, HTML/CSS/JS

## Project layout

- `backend/` — FastAPI API + OpenCV / ML routes
- `frontend/` — styles.css, app.js, UI markup
- `ml/` — ResNet18, Grad-CAM, training scripts
- `models/` — weights (`*.pth`); train locally, not in Git

**UI note:** HTML is in `frontend/page.txt`. For local run, copy or rename to `frontend/index.html`.

## Setup

```powershell
git clone https://github.com/SRAVS2003/Medical-Image-Analyzer.git
cd Medical-Image-Analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Model weights (Tab C)

```powershell
python ml/train_demo.py
```

Optional — [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) in `data/chest_xray/`:

```powershell
python ml/train_chest_xray.py --data-dir data/chest_xray --epochs 5
```

## Run

```powershell
copy frontend\page.txt frontend\index.html
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

## Disclaimer

Research and learning demo only — **not for clinical diagnosis or medical decisions.**

## Author

[SRAVS2003](https://github.com/SRAVS2003)
