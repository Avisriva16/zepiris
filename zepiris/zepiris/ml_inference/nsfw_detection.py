"""NSFW content detection service using MobileNetV2."""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.schemas.ml_inference import NSFWDetectionResult

INPUT_SIZE = (224, 224)


class NSFWDetectionService(ModelService):
    """NSFW detection service using MobileNetV2 binary classifier.

    Architecture: MobileNetV2 with 2-class linear head.
    Input: RGB image resized to 224x224, converted to tensor [0, 1].
    Output: class 1 probability (NSFW) via log-softmax.
    Model state_dict downloaded from Hugging Face Hub.
    """

    def __init__(
        self,
        huggingface_repo_id: str,
        huggingface_model_file: str = "model.pth",
        local_model_path: str | None = None,
        model_source: str = "auto",
        nsfw_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize NSFW detection service.

        Args:
            huggingface_repo_id: Hugging Face repository ID (e.g., "user/nsfw-detection")
            huggingface_model_file: Filename of model weights in the repo
            local_model_path: Optional local path to model weights
            model_source: ``"auto"``, ``"local"``, or ``"huggingface"``
            nsfw_threshold: Threshold for NSFW detection (default 0.5)
            device: Inference device ("cpu" or "cuda")
        """
        self._nsfw_threshold = nsfw_threshold
        self._model: nn.Module | None = None
        config = ModelServiceConfig(
            model_name="nsfw_detection",
            huggingface_repo_id=huggingface_repo_id,
            huggingface_model_file=huggingface_model_file,
            local_model_path=local_model_path,
            model_source=model_source,
            device=device,
        )
        super().__init__(config)
        self._transform = transforms.Compose(
            [
                transforms.Resize(INPUT_SIZE),
                transforms.ToTensor(),
            ]
        )

    def load_model(self) -> nn.Module:
        """Download state_dict from Hugging Face Hub and load into MobileNetV2.

        Returns:
            nn.Module: MobileNetV2 model in eval mode on configured device
        """
        if self._model is not None:
            return self._model

        model_bytes = self._download_model()
        buffer = io.BytesIO(model_bytes)

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
        model.load_state_dict(
            torch.load(buffer, map_location=self.config.device, weights_only=False)
        )
        model.to(self.config.device)
        model.eval()

        self._model = model
        return self._model

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Resize to 224x224 and convert to [0, 1] tensor via torchvision transforms.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            dict: {"tensor": torch.Tensor of shape (1, 3, 224, 224)}
        """
        pil_image = Image.fromarray(image_rgb).convert("RGB")
        tensor = self._transform(pil_image).unsqueeze(0)
        return {"tensor": tensor}

    def predict(self, preprocessed_data: dict) -> np.ndarray:
        """Run NSFW model forward pass and compute NSFW class probability.

        Args:
            preprocessed_data: dict with "tensor" key from preprocess()

        Returns:
            np.ndarray: [nsfw_probability], shape (1,)
        """
        model = self.load_model()
        tensor = preprocessed_data["tensor"].to(self.config.device)
        with torch.no_grad():
            logits = model(tensor).squeeze()
        prob_nsfw = torch.exp(logits[1] - torch.logsumexp(logits, dim=0)).item()
        return np.array([prob_nsfw], dtype=np.float32)

    def postprocess(self, output: np.ndarray) -> NSFWDetectionResult:
        """Convert NSFW probability to NSFWDetectionResult.

        Args:
            output: np.ndarray with NSFW probability, shape (1,)

        Returns:
            NSFWDetectionResult: is_safe=True when nsfw_prob <= threshold
        """
        nsfw_prob = float(output[0])
        is_safe = nsfw_prob <= self._nsfw_threshold
        probability = 1.0 - nsfw_prob

        return NSFWDetectionResult(is_safe=is_safe, probability=probability)
