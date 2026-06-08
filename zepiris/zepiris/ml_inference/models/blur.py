"""ResNet18-based binary blur detector architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ResNetBlurDetector(nn.Module):
    """ResNet18-based binary blur detector.

    Single sigmoid output: probability that the image is blurry.
    """

    def __init__(self, dropout_prob: float = 0.5) -> None:
        super().__init__()
        self.base_model = models.resnet18(weights=None)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning blur probability after sigmoid.

        Args:
            x: Input tensor, shape (B, 3, 224, 224)

        Returns:
            torch.Tensor: Blur probability, shape (B, 1), values in [0, 1]
        """
        return torch.sigmoid(self.base_model(x))
