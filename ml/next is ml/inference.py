from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.config import DEFAULT_WEIGHTS, IMG_SIZE, LABELS
from ml.gradcam import GradCAM, overlay_cam
from ml.model import load_classifier

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    gradcam_overlay_bgr: np.ndarray


def model_available() -> bool:
    return DEFAULT_WEIGHTS.is_file()


def _weights_path() -> Path:
    if not DEFAULT_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {DEFAULT_WEIGHTS}. "
            "Run: python ml/train_demo.py (from project root)"
        )
    return DEFAULT_WEIGHTS


@lru_cache(maxsize=1)
def _get_model():
    model = load_classifier(str(_weights_path()), num_classes=len(LABELS), device=str(_device))
    cam = GradCAM(model, model.layer4)
    return model, cam


def bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    pil_ready = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((pil_ready - mean) / std).unsqueeze(0).to(_device)


def predict_with_gradcam(bgr: np.ndarray) -> PredictionResult:
    model, cam_engine = _get_model()
    tensor = bgr_to_tensor(bgr)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    class_idx = int(probs.argmax())
    label = LABELS[class_idx]
    confidence = float(probs[class_idx])
    prob_map = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    model.train()
    cam = cam_engine(tensor, class_idx)
    model.eval()

    resized = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    overlay = overlay_cam(resized, cam)
    overlay = cv2.resize(overlay, (bgr.shape[1], bgr.shape[0]))

    return PredictionResult(
        label=label,
        confidence=confidence,
        probabilities=prob_map,
        gradcam_overlay_bgr=overlay,
    )
