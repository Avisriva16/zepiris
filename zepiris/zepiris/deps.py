from typing import Annotated

from fastapi import Depends, Request

from zepiris.config import Settings, get_settings
from zepiris.services.embedding import FaceEmbeddingProvider
from zepiris.services.iqa import MLInferenceIQAService
from zepiris.services.milvus_store import MilvusFaceStore
from zepiris.services.minio_storage import MinioStorageService


def settings_dep() -> Settings:
    return get_settings()


def minio_dep(request: Request) -> MinioStorageService:
    return request.app.state.minio_storage


def iqa_dep(request: Request) -> MLInferenceIQAService:
    return request.app.state.iqa


def embedding_dep(request: Request) -> FaceEmbeddingProvider:
    return request.app.state.embedding


def milvus_dep(request: Request) -> MilvusFaceStore:
    return request.app.state.milvus


SettingsDep = Annotated[Settings, Depends(settings_dep)]
MinioDep = Annotated[MinioStorageService, Depends(minio_dep)]
IQADep = Annotated[MLInferenceIQAService, Depends(iqa_dep)]
EmbeddingDep = Annotated[FaceEmbeddingProvider, Depends(embedding_dep)]
MilvusDep = Annotated[MilvusFaceStore, Depends(milvus_dep)]
