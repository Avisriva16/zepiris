from __future__ import annotations

import uuid
from io import BytesIO

from minio import Minio


class MinioStorageService:
    def __init__(
        self,
        client: Minio,
        bucket: str,
    ) -> None:
        self._client = client
        self._bucket = bucket

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def put_image(self, data: bytes, content_type: str = "application/octet-stream") -> str:
        object_key = f"faces/{uuid.uuid4().hex}"
        self._client.put_object(
            self._bucket,
            object_key,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_key

    def get_object_bytes(self, object_key: str) -> bytes:
        response = self._client.get_object(self._bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
