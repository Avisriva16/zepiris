"""Explicit model weight loading: local path, Hugging Face Hub, or automatic fallback."""

from __future__ import annotations

from pathlib import Path

# Env / config values: "auto" | "local" | "huggingface"


def _normalize_model_source(raw: str | None) -> str:
    s = (raw or "auto").strip().lower()
    if s not in ("auto", "local", "huggingface"):
        raise ValueError(f"model_source must be 'auto', 'local', or 'huggingface'; got {raw!r}")
    return s


def load_model_weights_bytes(
    *,
    model_name: str,
    model_source: str | None,
    local_model_path: str | None,
    huggingface_repo_id: str,
    huggingface_model_file: str,
) -> bytes:
    """Load raw model file bytes according to ``model_source``."""

    mode = _normalize_model_source(model_source)
    local_raw = (local_model_path or "").strip()
    local_path = Path(local_raw) if local_raw else None
    repo = (huggingface_repo_id or "").strip()

    if mode == "huggingface":
        return _load_from_huggingface(
            model_name=model_name,
            repo_id=repo,
            filename=huggingface_model_file,
        )

    if mode == "local":
        if not local_path:
            raise ValueError(
                f"model_source is local but local_model_path is empty (model={model_name})"
            )
        if not local_path.is_file():
            raise FileNotFoundError(
                f"model_source is local but file not found: {local_path} (model={model_name})"
            )
        return _read_local_file(local_path)

    # auto
    if local_path and local_path.is_file():
        return _read_local_file(local_path)
    if repo:
        return _load_from_huggingface(
            model_name=model_name,
            repo_id=repo,
            filename=huggingface_model_file,
        )
    if local_raw:
        raise FileNotFoundError(
            f"model_source is auto: no file at {local_path} and "
            f"huggingface_repo_id is empty (model={model_name})"
        )
    raise ValueError(
        f"model_source is auto: set local_model_path or huggingface_repo_id (model={model_name})"
    )


def _read_local_file(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def _load_from_huggingface(
    *,
    model_name: str,
    repo_id: str,
    filename: str,
) -> bytes:
    if not repo_id:
        raise ValueError(f"huggingface_repo_id is required for this load mode (model={model_name})")
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(model_path, "rb") as f:
        return f.read()
