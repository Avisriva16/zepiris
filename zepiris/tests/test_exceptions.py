"""Domain exception defaults."""

from zepiris.exceptions import EmptyUploadError, ZepirisServiceError


def test_zepiris_service_error_base() -> None:
    e = ZepirisServiceError("x", detail="y")
    assert e.status_code == 500
    assert e.detail == "y"


def test_empty_upload_error() -> None:
    e = EmptyUploadError()
    assert e.status_code == 400
    assert e.detail == "empty_upload"
