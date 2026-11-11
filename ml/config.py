from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "chest_xray_demo.pth"
LABELS = ("NORMAL", "PNEUMONIA")
IMG_SIZE = 224
