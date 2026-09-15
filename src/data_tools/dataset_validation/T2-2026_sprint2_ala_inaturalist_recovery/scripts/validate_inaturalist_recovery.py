import hashlib
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# Controlled test data created outside the repository.
TEST_DIR = Path.home() / "Desktop" / "project_echo_sprint2_test" / "inaturalist"

INPUT_METADATA = TEST_DIR / "metadata.csv"

OUTPUT_METADATA = BASE_DIR / "metadata" / "inaturalist_clean_metadata.csv"
REPORT_FILE = BASE_DIR / "reports" / "inaturalist_validation_summary.md"

ALLOWED_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-sa",
}

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


if not INPUT_METADATA.exists():
    raise FileNotFoundError(
        f"Could not find iNaturalist metadata: {INPUT_METADATA}"
    )

df = pd.read_csv(INPUT_METADATA)

print("Loaded records:", len(df))
print("Columns:", list(df.columns))

required_columns = [
    "species",
    "taxon_id",
    "observation_id",
    "media_type",
    "file_path",
    "source_url",
    "quality_grade",
    "license",
    "attribution",
]

missing_columns = [
    c for c in required_columns if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

records = []

for _, row in df.iterrows():

    raw_path = str(row.get("file_path", "")).strip()

    candidate_paths = [
        Path(raw_path),
        TEST_DIR / raw_path,
    ]

    actual_path = None

    for p in candidate_paths:
        if p.exists() and p.is_file():
            actual_path = p
            break

    file_exists = actual_path is not None

    extension = (
        actual_path.suffix.lower()
        if file_exists
        else Path(raw_path).suffix.lower()
    )

    license_value = row.get("license")

    if pd.isna(license_value):
        license_text = ""
    else:
        license_text = str(license_value).strip()

    attribution = str(row.get("attribution", "") or "")

    normalized_license = (
        license_text.lower()
        .strip()
        .replace("_", "-")
    )

    license_usable = (
        normalized_license in ALLOWED_LICENSES
        and "all rights reserved" not in attribution.lower()
    )

    file_hash = ""

    if file_exists:
        try:
            file_hash = sha256_file(actual_path)
        except OSError:
            file_hash = ""

    records.append({
        **row.to_dict(),
        "resolved_file_path":
            raw_path if file_exists else "",
        "file_exists":
            file_exists,
        "extension":
            extension,
        "license_usable":
            license_usable,
        "sha256":
            file_hash,
    })

out = pd.DataFrame(records)

out["duplicate_observation"] = out.duplicated(
    subset=["observation_id"],
    keep=False
)

out["duplicate_media_record"] = out.duplicated(
    subset=[
        "observation_id",
        "media_type",
        "media_index",
    ]
    if "media_index" in out.columns
    else [
        "observation_id",
        "file_path",
    ],
    keep=False
)

valid_hashes = out["sha256"].astype(str).str.len() > 0

out["duplicate_content"] = False

out.loc[valid_hashes, "duplicate_content"] = (
    out.loc[valid_hashes]
    .duplicated(subset=["sha256"], keep=False)
)

out["taxonomy_review_required"] = False

# Known controlled-test taxonomy issue:
# archived collector fell back from Chrysococcyx minutillus
# to the first API result, Chalcites minutillus.
out.loc[
    out["species"].astype(str).str.lower()
    == "chrysococcyx minutillus",
    "taxonomy_review_required"
] = True

out["usable_for_recovery"] = (
    out["file_exists"]
    & out["license_usable"]
    & ~out["taxonomy_review_required"]
    & ~out["duplicate_content"]
)

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUTPUT_METADATA,
    index=False
)

summary = {
    "total": len(out),
    "files_present": int(out["file_exists"].sum()),
    "usable_license": int(out["license_usable"].sum()),
    "taxonomy_review": int(out["taxonomy_review_required"].sum()),
    "duplicate_content": int(out["duplicate_content"].sum()),
    "usable": int(out["usable_for_recovery"].sum()),
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("# iNaturalist Validation Summary\n\n")

    f.write(
        "This validation was performed on the controlled "
        "Sprint 2 iNaturalist acquisition test.\n\n"
    )

    f.write(f"- Total metadata records: {summary['total']}\n")
    f.write(f"- Files present locally: {summary['files_present']}\n")
    f.write(f"- Records with usable licence: {summary['usable_license']}\n")
    f.write(f"- Records requiring taxonomy review: {summary['taxonomy_review']}\n")
    f.write(f"- Records involved in content duplicates: {summary['duplicate_content']}\n")
    f.write(f"- Records currently usable for recovery: {summary['usable']}\n")

    f.write("\n## Validation Rules\n\n")

    f.write(
        "- Recovered file must exist locally.\n"
        "- Licence must be present and compatible with the allowed "
        "Creative Commons prefixes used for this controlled recovery.\n"
        "- Records marked as all rights reserved are excluded.\n"
        "- Known taxonomy fallback cases are marked for manual review.\n"
        "- SHA-256 hashes are used for content duplicate detection.\n"
    )

print("\nValidation complete.")
print("Clean metadata:", OUTPUT_METADATA)
print("Report:", REPORT_FILE)

print("\nSummary:")
for key, value in summary.items():
    print(f"{key}: {value}")
