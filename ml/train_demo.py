"""
Train a small demo classifier (ResNet18) on bundled sample images.
For real metrics, use ml/train_chest_xray.py with the Kaggle pneumonia dataset.

Run from project root:
  python ml/train_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import DEFAULT_WEIGHTS, IMG_SIZE, LABELS, MODELS_DIR  # noqa: E402
from ml.model import build_classifier  # noqa: E402

SAMPLES = ROOT / "sample-images"


class SampleDataset(Dataset):
    def __init__(self) -> None:
        self.items: list[tuple[Path, int]] = [
            (SAMPLES / "sample_chest_xray_demo.png", 0),
            (SAMPLES / "sample_low_contrast_xray.png", 0),
            (SAMPLES / "sample_blobs_demo.png", 1),
        ]
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.92, 1.08)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items) * 40

    def __getitem__(self, idx: int):
        path, label = self.items[idx % len(self.items)]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def main() -> None:
    if not SAMPLES.is_dir():
        raise SystemExit(f"Missing sample-images folder: {SAMPLES}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(len(LABELS)).to(device)
    loader = DataLoader(SampleDataset(), batch_size=8, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(6):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch {epoch + 1}/6  loss={total / len(loader):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEFAULT_WEIGHTS)
    print(f"Saved demo weights to {DEFAULT_WEIGHTS}")
    print("Labels:", LABELS)


if __name__ == "__main__":
    main()
