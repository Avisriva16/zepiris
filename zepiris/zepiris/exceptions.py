"""Domain exceptions for the main API.

Raised from routes and services; translated to JSON by ``register_exception_handlers``
in :mod:`zepiris.exception_handlers`. Prefer these over raw ``HTTPException`` in
application code so status codes and payloads stay consistent.
"""

from __future__ import annotations

from typing import Any


class ZepirisServiceError(Exception):
    """Base class for expected API failures (4xx/5xx with a stable ``detail`` shape)."""

    default_status_code: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        detail: str | dict[str, Any] | list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code if status_code is not None else self.default_status_code
        self.detail: str | dict[str, Any] | list[Any] = (
            detail if detail is not None else (message or "error")
        )


# --- Client / upload validation ---


class EmptyUploadError(ZepirisServiceError):
    default_status_code = 400

    def __init__(self) -> None:
        super().__init__("empty_upload", detail="empty_upload")


class ImageTooLargeError(ZepirisServiceError):
    default_status_code = 422

    def __init__(self, mb: float, max_mb: float) -> None:
        d = f"image_too_large_{mb:.2f}MB_max_{max_mb}MB"
        super().__init__(d, detail=d)


class ImageEncodeError(ZepirisServiceError):
    default_status_code = 400

    def __init__(self) -> None:
        super().__init__("image_encode_failed", detail="image_encode_failed")


class EmbeddingDimensionMismatchError(ZepirisServiceError):
    """ML embedding length does not match Milvus collection dimension (config bug)."""

    default_status_code = 500

    def __init__(self, expected: int, got: int) -> None:
        d = f"embedding_dim_mismatch_expected_{expected}_got_{got}"
        super().__init__(d, detail=d)


# --- ML inference (HTTP client) ---


class MLInferenceUpstreamError(ZepirisServiceError):
    """ML microservice returned a non-success HTTP status."""

    default_status_code = 502

    def __init__(
        self,
        *,
        status_code: int,
        detail: dict[str, Any],
    ) -> None:
        super().__init__(
            str(detail.get("message", "ml_inference_request_failed")),
            status_code=status_code,
            detail=detail,
        )


class MLInferenceTimeoutError(ZepirisServiceError):
    default_status_code = 503

    def __init__(self) -> None:
        super().__init__(
            "ml_inference_timeout",
            detail="ml_inference_request_timeout",
        )


class MLInferenceTransportError(ZepirisServiceError):
    """Network / connection error talking to the ML microservice."""

    default_status_code = 503

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            detail={"message": "ml_inference_unreachable", "reason": message},
        )


# --- Face / vector store ---


class ImageQualityCheckFailedError(ZepirisServiceError):
    default_status_code = 422

    def __init__(self, detail: dict[str, Any]) -> None:
        super().__init__("image_quality_check_failed", detail=detail)


class DuplicateFaceIdError(ZepirisServiceError):
    default_status_code = 409

    def __init__(self, face_id: str) -> None:
        d = f"face_id_{face_id}_already_exists"
        super().__init__(d, detail=d)


class FaceRecordNotFoundError(ZepirisServiceError):
    default_status_code = 404

    def __init__(self, face_id: str) -> None:
        d = f"face_id_{face_id}_not_found"
        super().__init__(d, detail=d)


class MilvusOperationError(ZepirisServiceError):
    default_status_code = 500

    def __init__(self, operation: str, cause: BaseException) -> None:
        d = f"milvus_{operation}_failed: {cause}"
        super().__init__(d, detail=d)
