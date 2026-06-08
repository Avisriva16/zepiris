"""MobileNetV3-Large binary classifier for spoof detection architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3LSpoof(nn.Module):
    """MobileNetV3-Large binary classifier for spoof detection.

    Single logit output, use sigmoid for probability.
    Trained with BCEWithLogitsLoss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        self.mobilenet.classifier[-1] = nn.Linear(in_features=1280, out_features=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logit.

        Args:
            x: Input tensor, shape (B, 3, 224, 224), ImageNet-normalized

        Returns:
            torch.Tensor: Raw logit, shape (B, 1)
        """
        return self.mobilenet(x)
