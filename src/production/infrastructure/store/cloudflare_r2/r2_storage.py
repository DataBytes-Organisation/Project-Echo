"""Minimal S3-compatible client for a private Cloudflare R2 bucket."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator


@dataclass(frozen=True)
class R2Config:
    account_id: str
    bucket_name: str
    access_key_id: str
    secret_access_key: str
    dataset_prefix: str = "prototype"

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_env(cls) -> "R2Config":
        required_names = (
            "R2_ACCOUNT_ID",
            "R2_BUCKET_NAME",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
        )
        values = {
            name: os.environ.get(name, "").strip() for name in required_names
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(
                "Missing required Cloudflare R2 configuration: "
                + ", ".join(missing)
            )

        return cls(
            account_id=values["R2_ACCOUNT_ID"],
            bucket_name=values["R2_BUCKET_NAME"],
            access_key_id=values["R2_ACCESS_KEY_ID"],
            secret_access_key=values["R2_SECRET_ACCESS_KEY"],
            dataset_prefix=(
                os.environ.get("R2_DATASET_PREFIX", "prototype").strip()
                or "prototype"
            ),
        )


class R2Storage:
    """Storage operations shared by the prototype engine and simulator."""

    def __init__(self, config: R2Config, client: Any | None = None) -> None:
        self.config = config
        self._client = client or self._create_client(config)

    @staticmethod
    def _create_client(config: R2Config) -> Any:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise RuntimeError(
                "Cloudflare R2 support requires boto3. Install requirements.txt."
            ) from error

        return boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name="auto",
            config=Config(retries={"max_attempts": 3, "mode": "standard"}),
        )

    def check_connection(self) -> None:
        self._client.head_bucket(Bucket=self.config.bucket_name)

    def iter_object_keys(self, prefix: str = "") -> Iterator[str]:
        continuation_token: str | None = None
        while True:
            request = {"Bucket": self.config.bucket_name, "Prefix": prefix}
            if continuation_token:
                request["ContinuationToken"] = continuation_token

            response = self._client.list_objects_v2(**request)
            for item in response.get("Contents", []):
                key = item.get("Key")
                if key and not key.endswith("/"):
                    yield key

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                raise RuntimeError(
                    "R2 returned a truncated listing without a continuation token"
                )

    def group_objects_by_species(self, prefix: str | None = None) -> dict[str, list[str]]:
        """Group file keys using the dataset's first directory as species."""
        normalized = (self.config.dataset_prefix if prefix is None else prefix).strip("/")
        key_prefix = f"{normalized}/" if normalized else ""
        grouped: dict[str, list[str]] = {}

        for key in self.iter_object_keys(key_prefix):
            relative_key = key[len(key_prefix) :]
            parts = relative_key.split("/", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            grouped.setdefault(parts[0], []).append(key)

        for keys in grouped.values():
            keys.sort()
        return dict(sorted(grouped.items()))

    def list_species(self, prefix: str | None = None) -> list[str]:
        return list(self.group_objects_by_species(prefix))

    def download_bytes(self, key: str) -> bytes:
        if not key or key.endswith("/"):
            raise ValueError("R2 object key must identify a file")
        response = self._client.get_object(
            Bucket=self.config.bucket_name,
            Key=key,
        )
        return response["Body"].read()

    def download_file(self, key: str, destination: str | Path) -> Path:
        if not key or key.endswith("/"):
            raise ValueError("R2 object key must identify a file")
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(
            self.config.bucket_name,
            key,
            str(destination_path),
        )
        return destination_path

    def upload_file(
        self,
        source: str | Path | BinaryIO,
        key: str,
        content_type: str | None = None,
    ) -> None:
        if not key or key.endswith("/"):
            raise ValueError("R2 object key must identify a file")
        extra_args = {"ContentType": content_type} if content_type else None
        kwargs = {"ExtraArgs": extra_args} if extra_args else {}

        if hasattr(source, "read"):
            self._client.upload_fileobj(
                source, self.config.bucket_name, key, **kwargs
            )
        else:
            self._client.upload_file(
                str(source), self.config.bucket_name, key, **kwargs
            )
