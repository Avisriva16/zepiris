from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from zepiris.schemas.ml_inference import ImageQualityAssessmentResult

MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# SEARCH_STRUCT  – vector-similarity results
# ---------------------------------------------------------------------------


class SearchMatch(BaseModel):
    """A single vector-search hit."""

    id: str
    score: float


class SearchStruct(BaseModel):
    """Maps to SEARCH_STRUCT in the design doc."""

    matches: list[SearchMatch] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CRUD_RESULT  – outcome of insert / upsert / delete
# ---------------------------------------------------------------------------

CRUDOperation = Literal["INSERT", "UPSERT", "DELETE"]
CRUDStatus = Literal["success", "failed", "not_found"]


class CRUDResult(BaseModel):
    """Maps to CRUD_RESULT in the design doc."""

    operation: CRUDOperation
    status: CRUDStatus


# ---------------------------------------------------------------------------
# API responses  (camelCase aliases match the design-doc JSON contracts)
# ---------------------------------------------------------------------------


class SearchResponse(BaseModel):
    """Response for the Search (1-to-N) endpoint."""

    request_id: str = Field(..., alias="requestId")
    image_quality_assessment: ImageQualityAssessmentResult = Field(
        ..., alias="imageQualityAssessment"
    )
    search_result: SearchStruct = Field(..., alias="searchResult")

    model_config = {"populate_by_name": True}


class UpsertResponse(BaseModel):
    """Response for the Insert / Upsert endpoint."""

    request_id: str = Field(..., alias="requestId")
    image_quality_assessment: ImageQualityAssessmentResult = Field(
        ..., alias="imageQualityAssessment"
    )
    user_operation_result: CRUDResult = Field(..., alias="userOperationResult")

    model_config = {"populate_by_name": True}


class DeleteResponse(BaseModel):
    """Response for the Delete endpoint."""

    request_id: str = Field(..., alias="requestId")
    user_operation_result: CRUDResult = Field(..., alias="userOperationResult")

    model_config = {"populate_by_name": True}
