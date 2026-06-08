import os
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zepiris.version import __version__ as _package_version


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZEPIRIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "ZepIris"
    api_version: str = _package_version
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    minio_endpoint: str = Field(default="localhost:9002", description="host:port")
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "zepiris"
    minio_secure: bool = False

    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "zepiris_faces"
    milvus_embedding_dim: int = 512
    milvus_search_threshold: float = 0.5

    #: Required. ML inference base URL; IQA uses POST /v1/iqa/assess, embeddings POST /v1/face/embed.
    #: Also accepts legacy env ML_INFERENCE_SERVICE_URL (no ZEPIRIS_ prefix) if this is unset.
    ml_inference_service_url: str = Field(default="", description="e.g. http://ml-inference:8001")

    @model_validator(mode="after")
    def _require_ml_inference_url(self) -> "Settings":
        u = (self.ml_inference_service_url or "").strip()
        if not u:
            u = (os.environ.get("ML_INFERENCE_SERVICE_URL") or "").strip()
        if not u:
            raise ValueError(
                "ZEPIRIS_ML_INFERENCE_SERVICE_URL is required "
                "(e.g. http://ml-inference:8001 or http://localhost:8001). "
                "Alternatively set ML_INFERENCE_SERVICE_URL."
            )
        self.ml_inference_service_url = u
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
