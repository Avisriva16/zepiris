"""FastAPI exception registration for the main API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from zepiris.exceptions import ZepirisServiceError

logger = logging.getLogger("zepiris")


def _detail_payload(detail: str | dict[str, Any] | list[Any]) -> dict[str, Any]:
    return {"detail": detail}


def register_exception_handlers(app: FastAPI) -> None:
    """Register domain and fallback handlers. FastAPI's ``HTTPException`` handler stays in effect."""

    @app.exception_handler(ZepirisServiceError)
    async def zepiris_service_error_handler(
        _request: Request,
        exc: ZepirisServiceError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_detail_payload(exc.detail),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "Unhandled exception path=%s",
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )
