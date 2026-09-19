"""
Project Echo
Sprint 2 - ALA Dataset Acquisition Recovery

Controlled recovery of sound records from:
The Glenelg-Hopkins Soundscape Project - Frog Calls (dr23984)

Recovered ALA data is intentionally stored separately from the
Sprint 2 balanced dataset and the iNaturalist recovery.
"""

import csv
import hashlib
import os
import time
import requests

SEARCH_URL = "https://biocache.ala.org.au/ws/occurrences/search"
MEDIA_URL = "https://api.ala.org.au/images/image/{}/original"

RESOURCE_UID = "dr23984"

# Controlled recovery only.
MAX_RECORDS = 5

TIMEOUT = 30
DELAY = 0.5

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "recovered_data",
    "ala"
)

METADATA_DIR = os.path.join(
    BASE_DIR,
    "metadata"
)

METADATA_FILE = os.path.join(
    METADATA_DIR,
    "ala_recovered_metadata.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)


def get_extension(content_type):
    """Return an appropriate extension for supported audio MIME types."""

    content_type = (content_type or "").lower()

    if "flac" in content_type:
        return ".flac"

    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"

    if "wav" in content_type or "wave" in content_type:
        return ".wav"

    if "ogg" in content_type:
        return ".ogg"

    if "mp4" in content_type or "m4a" in content_type:
        return ".m4a"

    return None


def sha256_bytes(content):
    """Return SHA-256 hash for downloaded content."""
    return hashlib.sha256(content).hexdigest()


print("=" * 60)
print("ALA CONTROLLED RECOVERY")
print("=" * 60)

params = {
    "q": f"data_resource_uid:{RESOURCE_UID}",
    "fq": ['multimedia:"Sound"'],
    "pageSize": MAX_RECORDS,
}

print("\nQuerying ALA occurrence service...")

response = requests.get(
    SEARCH_URL,
    params=params,
    timeout=TIMEOUT
)

response.raise_for_status()

data = response.json()
records = data.get("occurrences", [])

print("Resource:", RESOURCE_UID)
print("Total sound records available:", data.get("totalRecords"))
print("Records requested:", MAX_RECORDS)
print("Records returned:", len(records))

rows = []
successful_downloads = 0
failed_downloads = 0

for record_number, record in enumerate(records, 1):

    occurrence_uuid = record.get("uuid")

    scientific_name = (
        record.get("scientificName")
        or record.get("species")
        or "unknown"
    )

    species = record.get("species") or scientific_name

    sound_ids = record.get("sounds") or []

    if not sound_ids:
        print(
            f"\n[{record_number}] "
            f"Skipping {occurrence_uuid}: no sound IDs."
        )
        continue

    for sound_index, sound_id in enumerate(sound_ids, 1):

        print("\n" + "-" * 60)
        print(f"[{record_number}] {scientific_name}")
        print("Occurrence UUID:", occurrence_uuid)
        print("Sound ID:", sound_id)
        print("License:", record.get("license"))

        media_url = MEDIA_URL.format(sound_id)

        try:
            audio_response = requests.get(
                media_url,
                timeout=TIMEOUT,
                allow_redirects=True
            )

            audio_response.raise_for_status()

            content_type = (
                audio_response.headers.get("content-type", "")
                .split(";")[0]
                .strip()
            )

            print("Content-Type:", content_type)

            extension = get_extension(content_type)

            # Do not save responses that are not recognised audio.
            if extension is None:
                print(
                    "SKIPPED: response is not a supported "
                    "audio content type."
                )
                failed_downloads += 1
                continue

            filename = (
                f"{occurrence_uuid}_{sound_index}{extension}"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            with open(output_path, "wb") as f:
                f.write(audio_response.content)

            file_hash = sha256_bytes(
                audio_response.content
            )

            file_size = len(audio_response.content)

            print("Saved:", output_path)
            print("Bytes:", file_size)
            print("SHA-256:", file_hash)

            rows.append({
                "source": "ALA",
                "data_resource_uid":
                    record.get("dataResourceUid"),
                "data_resource_name":
                    record.get("dataResourceName"),
                "occurrence_uuid":
                    occurrence_uuid,
                "scientific_name":
                    record.get("scientificName"),
                "species":
                    species,
                "basis_of_record":
                    record.get("basisOfRecord"),
                "latitude":
                    record.get("decimalLatitude"),
                "longitude":
                    record.get("decimalLongitude"),
                "license":
                    record.get("license"),
                "sound_id":
                    sound_id,
                "media_url":
                    media_url,
                "file_path":
                    os.path.relpath(
                        output_path,
                        BASE_DIR
                    ),
                "content_type":
                    content_type,
                "file_size_bytes":
                    file_size,
                "sha256":
                    file_hash,
            })

            successful_downloads += 1

        except requests.RequestException as error:
            print("DOWNLOAD FAILED:", error)
            failed_downloads += 1

        time.sleep(DELAY)


if rows:

    fieldnames = list(rows[0].keys())

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("RECOVERY COMPLETE")
    print("=" * 60)

    print("Successful downloads:", successful_downloads)
    print("Failed/skipped downloads:", failed_downloads)
    print("Metadata records:", len(rows))
    print("Metadata file:", METADATA_FILE)

else:

    print("\nNo valid audio files were recovered.")
