"""Face embedding service using InsightFace."""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

from zepiris.ml_inference.base import ModelService, ModelServiceConfig
from zepiris.schemas.ml_inference import FaceEmbeddingResult

SUPPORTED_FACE_MODELS = ("auraface", "buffalo_l")


class FaceEmbeddingService(ModelService):
    """Face embedding service using InsightFace.

    Supports AuraFace (default) and buffalo_l models.
    Uses InsightFace FaceAnalysis for detection + recognition.
    Selects the most central face that exceeds a minimum area threshold.
    Returns a 512-d normalized embedding vector.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        detection_size: tuple[int, int] = (640, 640),
        facial_area_threshold: float = 0.01,
        device: str = "cpu",
        face_model: str = "auraface",
    ) -> None:
        """Initialize face embedding service.

        Args:
            embedding_dim: Expected embedding dimension (default 512)
            detection_size: Face detection size as (width, height) tuple
            facial_area_threshold: Minimum face area as fraction of image area
            device: Inference device (for now only "cpu" is supported)
            face_model: Model to use from SUPPORTED_FACE_MODELS
        """
        if face_model not in SUPPORTED_FACE_MODELS:
            raise ValueError(
                f"Unsupported face model '{face_model}'. Choose from: {SUPPORTED_FACE_MODELS}"
            )
        self._embedding_dim = embedding_dim
        self._detection_size = detection_size
        self._facial_area_threshold = facial_area_threshold
        self._face_model = face_model
        self._face_app: FaceAnalysis | None = None
        config = ModelServiceConfig(
            model_name="face_embedding",
            device=device,
        )
        super().__init__(config)

    def _download_auraface(self) -> None:
        """Download AuraFace weights from HuggingFace Hub if not present."""
        from huggingface_hub import snapshot_download

        snapshot_download("fal/AuraFace-v1", local_dir="models/auraface")

    def load_model(self) -> FaceAnalysis:
        """Initialize InsightFace FaceAnalysis with the configured model.

        Returns:
            FaceAnalysis: Prepared model with detection + recognition only
        """
        if self._face_app is not None:
            return self._face_app

        ctx_id = 0 if self.config.device != "cpu" else -1

        if self._face_model == "auraface":
            self._download_auraface()
            app = FaceAnalysis(
                name="auraface",
                root=".",
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
        elif self._face_model == "buffalo_l":
            app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
        else:
            raise ValueError(
                f"Unsupported face model '{self._face_model}'. Choose from: {SUPPORTED_FACE_MODELS}"
            )

        app.prepare(ctx_id=ctx_id, det_size=self._detection_size)

        self._face_app = app
        return self._face_app

    def preprocess(self, image_rgb: np.ndarray) -> dict:
        """Detect faces and select the most central face above area threshold.

        Args:
            image_rgb: Input image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            dict: {"image": original image, "face": selected Face object or None}
        """
        app = self.load_model()

        bboxes, kpss = app.det_model.detect(image_rgb, max_num=0, metric="default")

        if len(bboxes) == 0:
            return {"image": image_rgb, "face": None}

        h, w = image_rgb.shape[:2]
        img_area = h * w
        img_center = np.array([w / 2.0, h / 2.0])

        filtered_indices = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box[:4]
            face_area = (x2 - x1) * (y2 - y1)
            if face_area / img_area > self._facial_area_threshold:
                filtered_indices.append(i)

        candidates = filtered_indices if filtered_indices else list(range(len(bboxes)))

        selected_idx = 0
        min_dist = float("inf")
        for i in candidates:
            x1, y1, x2, y2 = bboxes[i][:4]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            dist = np.linalg.norm(center - img_center)
            if dist < min_dist:
                min_dist = dist
                selected_idx = i

        face = Face(
            bbox=bboxes[selected_idx, :4],
            kps=kpss[selected_idx],
            det_score=bboxes[selected_idx, 4],
        )

        return {"image": image_rgb, "face": face}

    def predict(self, preprocessed_data: dict) -> tuple[np.ndarray, bool]:
        """Extract embedding for the detected face using the recognition model.

        Args:
            preprocessed_data: dict with "image" and "face" keys from preprocess()

        Returns:
            tuple[np.ndarray, bool]: Face embedding (shape (512,), dtype float32)
                                     and face_detected flag (True if face was found)
        """
        face = preprocessed_data["face"]
        if face is None:
            return np.zeros(self._embedding_dim, dtype=np.float32), False

        app = self.load_model()
        embedding = app.models["recognition"].get(preprocessed_data["image"], face)

        return np.asarray(embedding, dtype=np.float32), True

    def postprocess(self, output: tuple[np.ndarray, bool]) -> FaceEmbeddingResult:
        """L2-normalize embedding and wrap in result schema.

        Args:
            output: Tuple of (embedding, face_detected) from predict()

        Returns:
            FaceEmbeddingResult: Normalized embedding with face detection status and metadata
        """
        embedding, face_detected = output

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        embedding_list = embedding.astype(np.float32).tolist()
        return FaceEmbeddingResult(
            face_detected=face_detected,
            embedding=embedding_list,
            embedding_dim=len(embedding_list),
        )

    def embed(self, image_rgb: np.ndarray) -> FaceEmbeddingResult:
        """Generate face embedding from image.

        Convenience alias for ``forward()``.

        Args:
            image_rgb: Input face image in RGB format, shape (H, W, 3), dtype uint8

        Returns:
            FaceEmbeddingResult: L2-normalized embedding vector with face detection status and metadata
        """
        return self.forward(image_rgb)
