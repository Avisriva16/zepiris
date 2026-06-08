"""Blur detection service using ResNet18-based binary classifier."""

from __future__ import annotations

import io

import cv2
import numpy as np
import torch

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.ml_inference.models import ResNetBlurDetector
from zepiris.schemas.ml_inference import BlurDetectionResult

INPUT_SIZE = (224, 224)


class BlurDetectionService(ModelService):
    """Blur detection service using ResNet18 binary classifier.

    Architecture: ResNet18 with dropout + linear head → sigmoid.
    Input: RGB image resized to 224x224, normalized to [0, 1].
    Output: probability of blur; > 0.5 means blurry.
    Model weights are downloaded from Hugging Face Hub.
    """

    def __init__(
        self,
        huggingface_repo_id: str,
        huggingface_model_file: str = "model.pth",
        local_model_path: str | None = None,
        model_source: str = "auto",
        blur_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize blur detection service.

        Args:
            huggingface_repo_id: Hugging Face repository ID (e.g., "user/blur-detection")
            huggingface_model_file: Filename of model weights in the repo
            local_model_path: Optional local path to model weights
            model_source: ``"auto"``, ``"local"``, or ``"huggingface"``
            blur_threshold: Threshold for blur detection (default 0.5)
            device: Inference device ("cpu" or "cuda")
        """
        self._blur_threshold = blur_threshold
        self._model: ResNetBlurDetector | None = None
        config = ModelServiceConfig(
            model_name="blur_detection",
            huggingface_repo_id=huggingface_repo_id,
            huggingface_model_file=huggingface_model_file,
            local_model_path=local_model_path,
            model_source=model_source,
            device=device,
        )
        super().__init__(config)

    def load_model(self) -> ResNetBlurDetector:
        """Download state_dict from Hugging Face Hub and load into ResNetBlurDetector.

        Returns:
            ResNetBlurDetector: Model in eval mode on configured device
        """
        if self._model is not None:
            return self._model

        model_bytes = self._download_model()
        buffer = io.BytesIO(model_bytes)

        model = ResNetBlurDetector(dropout_prob=0.5)
        state_dict = torch.load(buffer, map_location=self.config.device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(self.config.device)
        model.eval()

        self._model = model
        return self._model

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Resize to 224x224, normalize to [0, 1], convert to NCHW tensor.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            dict: {"tensor": torch.Tensor of shape (1, 3, 224, 224)}
        """
        resized = cv2.resize(image_rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)
        chw = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0)
        return {"tensor": tensor}

    def predict(self, preprocessed_data: dict) -> np.ndarray:
        """Run blur model forward pass.

        Args:
            preprocessed_data: dict with "tensor" key from preprocess()

        Returns:
            np.ndarray: Blur probability, shape (1,)
        """
        model = self.load_model()
        tensor = preprocessed_data["tensor"].to(self.config.device)
        with torch.no_grad():
            output = model(tensor)
        return output.cpu().numpy().flatten()

    def postprocess(self, output: np.ndarray) -> BlurDetectionResult:
        """Convert blur probability to BlurDetectionResult.

        Args:
            output: np.ndarray with blur probability, shape (1,)

        Returns:
            BlurDetectionResult: is_sharp=True when blur_prob <= threshold
        """
        blur_prob = float(output[0])
        is_sharp = blur_prob <= self._blur_threshold
        probability = 1.0 - blur_prob

        return BlurDetectionResult(is_sharp=is_sharp, probability=probability)
