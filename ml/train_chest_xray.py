"""
Train pneumonia vs normal on Kaggle-style folder layout:

  data/chest_xray/train/NORMAL/*.jpeg
  data/chest_xray/train/PNEUMONIA/*.jpeg
  data/chest_xray/test/NORMAL/*.jpeg
  data/chest_xray/test/PNEUMONIA/*.jpeg

Run from project root:
  python ml/train_chest_xray.py --data-dir data/chest_xray --epochs 5

Writes models/chest_xray_demo.pth and prints test metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import DEFAULT_WEIGHTS, IMG_SIZE, MODELS_DIR  # noqa: E402
from ml.model import build_classifier  # noqa: E402


def make_loaders(data_dir: Path, batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = ImageFolder(data_dir / "train", transform=train_tf)
    test_ds = ImageFolder(data_dir / "test", transform=test_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
        train_ds.classes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "chest_xray")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not (args.data_dir / "train").is_dir():
        raise SystemExit(f"Train folder not found: {args.data_dir / 'train'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = make_loaders(args.data_dir, args.batch_size)
    model = build_classifier(len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.item())
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={running / len(train_loader):.4f}")

    model.eval()
    ys, ps, scores = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.extend(y.numpy().tolist())
            ps.extend(pred.tolist())
            scores.extend(prob.tolist())

    print(classification_report(ys, ps, target_names=classes))
    if len(set(ys)) > 1:
        print("ROC-AUC (class index 1):", round(roc_auc_score(ys, scores), 4))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEFAULT_WEIGHTS)
    print("Saved:", DEFAULT_WEIGHTS)


if __name__ == "__main__":
    main()
