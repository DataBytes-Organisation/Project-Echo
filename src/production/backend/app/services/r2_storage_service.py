"""Backend adapter for Cloudflare R2 object storage."""

from functools import lru_cache
from typing import Any, BinaryIO, Optional

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from app.config import settings


class R2StorageUnavailable(RuntimeError):
    """Raised when R2 cannot currently be reached or configured."""


class R2UploadFailed(RuntimeError):
    """Raised when R2 rejects or fails an upload operation."""

class R2DownloadFailed(RuntimeError):
    """Raised when R2 rejects or fails a download operation."""


class R2ObjectNotFound(RuntimeError):
    """Raised when the requested R2 object does not exist."""

@lru_cache(maxsize=1)
def get_r2_storage() -> Any:
    try:
        from cloudflare_r2 import R2Config, R2Storage
    except ImportError as exc:
        raise R2StorageUnavailable(
            "Cloudflare R2 storage package is unavailable"
        ) from exc

    try:
        config = R2Config.from_settings(settings)
        return R2Storage(config)
    except (ValueError, RuntimeError, BotoCoreError) as exc:
        raise R2StorageUnavailable(
            "Cloudflare R2 storage is unavailable"
        ) from exc


def upload_to_r2(
    storage: Any,
    source: BinaryIO,
    key: str,
    content_type: Optional[str] = None,
) -> None:
    try:
        storage.upload_file(
            source,
            key,
            content_type=content_type,
        )
    except (
        EndpointConnectionError,
        ConnectionClosedError,
        ConnectTimeoutError,
        ReadTimeoutError,
    ) as exc:
        raise R2StorageUnavailable(
            "Cloudflare R2 is temporarily unavailable"
        ) from exc
    except (ClientError, BotoCoreError) as exc:
        raise R2UploadFailed(
            "Cloudflare R2 upload failed"
        ) from exc
    except Exception as exc:
        raise R2UploadFailed(
            "Cloudflare R2 upload failed"
        ) from exc

def download_from_r2(
    storage: Any,
    key: str,
) -> bytes:
    try:
        return storage.download_bytes(key)

    except (
        EndpointConnectionError,
        ConnectionClosedError,
        ConnectTimeoutError,
        ReadTimeoutError,
    ) as exc:
        raise R2StorageUnavailable(
            "Cloudflare R2 is temporarily unavailable"
        ) from exc

    except ClientError as exc:
        error_code = str(
            exc.response.get("Error", {}).get("Code", "")
        )

        if error_code in {
            "NoSuchKey",
            "NotFound",
            "404",
        }:
            raise R2ObjectNotFound(
                "Cloudflare R2 object was not found"
            ) from exc

        raise R2DownloadFailed(
            "Cloudflare R2 download failed"
        ) from exc

    except BotoCoreError as exc:
        raise R2DownloadFailed(
            "Cloudflare R2 download failed"
        ) from exc

    except Exception as exc:
        raise R2DownloadFailed(
            "Cloudflare R2 download failed"
        ) from exc

def delete_from_r2_best_effort(storage: Any, key: str) -> bool:
    try:
        storage.delete_object(key)
        return True
    except Exception:
        return False
