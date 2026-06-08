from fastapi import APIRouter

from zepiris.deps import SettingsDep

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(settings: SettingsDep) -> dict[str, str]:
    """Lightweight readiness: config loaded. Extend with MinIO/Milvus pings if needed."""
    _ = settings
    return {"status": "ready"}
