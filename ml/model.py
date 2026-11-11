from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_classifier(num_classes: int = 2) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(weights_path: str, num_classes: int = 2, device: str | None = None) -> nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(num_classes)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
