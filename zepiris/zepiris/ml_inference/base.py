"""Base class for ML inference services with Hugging Face model download."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict


class ModelServiceConfig(BaseModel):
    """Configuration for an ML inference service.

    Attributes:
        model_name: Human-readable name (e.g., "nsfw_detection")
        huggingface_repo_id: Hugging Face model repository ID (e.g., "user/model-name")
        huggingface_model_file: Filename of model weights in the repo (e.g., "model.pth")
        local_model_path: Optional local filesystem path to model weights.
        model_source: ``"auto"`` (local file if present, else Hub), ``"local"``, or
            ``"huggingface"``.
        device: Inference device ("cpu" or "cuda")
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    huggingface_repo_id: str = ""
    huggingface_model_file: str = ""
    local_model_path: str | None = None
    model_source: str = "auto"
    device: str = "cpu"


class ModelService:
    """Base class for ML model inference services.

    Defines the preprocess → predict → postprocess pipeline with a concrete
    ``forward`` method.  Subclasses that load PyTorch weights from Hugging Face
    can reuse ``_download_model``; others (e.g. AuraFace OR Buffalo_l) override
    ``load_model`` entirely.

    Subclasses must implement:
        - load_model(): Load (and cache) the underlying model
        - preprocess(image_rgb): Preprocess image for inference
        - predict(preprocessed_data): Run model forward pass
        - postprocess(output): Convert output to Pydantic result schema
    """

    def __init__(self, config: ModelServiceConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> object:
        """Load (and cache) the underlying model. Called lazily on first use.

        Subclasses must override this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement load_model()")

    def _download_model(self) -> bytes:
        """Download or load model weights from configured source.

        Resolution is delegated to :func:`~zepiris.ml_inference.model_loading.load_model_weights_bytes`.

        Returns:
            bytes: Raw model file contents
        """
        from zepiris.ml_inference.model_loading import load_model_weights_bytes

        return load_model_weights_bytes(
            model_name=self.config.model_name,
            model_source=self.config.model_source,
            local_model_path=self.config.local_model_path,
            huggingface_repo_id=self.config.huggingface_repo_id,
            huggingface_model_file=self.config.huggingface_model_file,
        )

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Preprocess image for model inference.

        Subclasses must override this method.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            dict: Preprocessed data ready for model input
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement preprocess()")

    def predict(self, preprocessed_data: dict) -> np.ndarray:
        """Run model forward pass on preprocessed data.

        Subclasses must override this method.

        Args:
            preprocessed_data: Output from preprocess()

        Returns:
            np.ndarray: Raw model output
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement predict()")

    def postprocess(self, output: np.ndarray) -> BaseModel:
        """Convert raw model output to a Pydantic result schema.

        Subclasses must override this method.

        Args:
            output: Raw model output from predict()

        Returns:
            BaseModel: Pydantic schema with typed inference result
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement postprocess()")

    def forward(self, image_rgb: np.ndarray) -> BaseModel:
        """Run full inference pipeline: preprocess → predict → postprocess.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            BaseModel: Typed inference result
        """
        preprocessed = self.preprocess(image_rgb)
        output = self.predict(preprocessed)
        return self.postprocess(output)
