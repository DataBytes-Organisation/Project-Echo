import os

import boto3
from botocore.config import Config


class R2Storage:
    """Read-only access to the Project Echo dataset in Cloudflare R2."""

    def __init__(self):
        required = (
            "R2_ACCOUNT_ID",
            "R2_BUCKET_NAME",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
        )
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "Missing required R2 environment variables: " + ", ".join(missing)
            )

        self.bucket_name = os.environ["R2_BUCKET_NAME"]
        self.dataset_prefix = os.environ.get("R2_DATASET_PREFIX", "").strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=(
                f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
            ),
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    def list_species(self):
        prefix = f"{self.dataset_prefix}/" if self.dataset_prefix else ""
        species_names = set()
        paginator = self.client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                relative_key = item["Key"][len(prefix):]
                parts = relative_key.split("/", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    species_names.add(parts[0])

        if not species_names:
            raise RuntimeError("No species audio was found in the configured R2 prefix")

        return sorted(species_names)
