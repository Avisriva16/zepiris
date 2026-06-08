"""PyTorch model architecture definitions."""

from zepiris.ml_inference.models.blur import ResNetBlurDetector
from zepiris.ml_inference.models.spoof import MobileNetV3LSpoof

__all__ = [
    "ResNetBlurDetector",
    "MobileNetV3LSpoof",
]
