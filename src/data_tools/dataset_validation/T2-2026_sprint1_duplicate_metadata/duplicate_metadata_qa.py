"""
Task 3.13 - Dataset Duplicate and Metadata Consistency Validation
Sprint 1 | Workstream: Dataset and Evaluation
Author: Deepanshi Kathpal

This script focuses on:
    1. Exact duplicate file paths
    2. Duplicate metadata records
    3. File-hash based audio duplicate detection
    4. Species-name formatting / inconsistent identifiers
    5. Missing source / location / label information
    6. A single structured, combined issue list

Non-destructive:
    Nothing is deleted or modified. The script only analyses the manifest
    and/or dataset and produces CSV reports.
"""

from pathlib import Path
import zipfile
import hashlib
import re

import pandas as pd
from tqdm import tqdm


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path.cwd()

# Dataset configuration
DATASET_ZIP_NAME = "datasets.zip"
DATASET_FOLDER_NAME = "datasets"

ZIP_PATH = PROJECT_ROOT / DATASET_ZIP_NAME
DATASET_PATH = PROJECT_ROOT / DATASET_FOLDER_NAME

# Report output folder
REPORT_OUTPUT = PROJECT_ROOT / "reports"
REPORT_OUTPUT.mkdir(exist_ok=True)

# Existing manifest
EXISTING_MANIFEST_PATH = PROJECT_ROOT / "dataset_manifest.csv"

# True = use the existing manifest
# False = scan the physical dataset folder
USE_EXISTING_MANIFEST = True

# Hash configuration
HASH_ALGORITHM = "sha256"
HASH_CHUNK_SIZE = 8192


# ==========================================
# DATASET AVAILABILITY CHECK
# ==========================================

if not USE_EXISTING_MANIFEST:

    if not DATASET_PATH.exists():

        if not ZIP_PATH.exists():

            raise FileNotFoundError(
                f"No dataset found. Expected either:\n"
                f"1. Folder: {DATASET_PATH}\n"
                f"2. ZIP file: {ZIP_PATH}"
            )

        print("Extracting dataset ZIP...")

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(PROJECT_ROOT)

        print(f"Dataset extracted to: {DATASET_PATH}")


# ==========================================
# LABEL NORMALISATION
# ==========================================

def normalise_label(name):
    """
    Converts species names into a consistent identifier.

    Examples:
        'Eastern-Rosella' -> 'eastern_rosella'
        ' eastern rosella ' -> 'eastern_rosella'
        'Eastern Rosella' -> 'eastern_rosella'
    """

    name = str(name).lower().strip()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    name = re.sub(
        r"_+",
        "_",
        name
    )

    return name.strip("_")


# ==========================================
# DATASET SCANNER
# ==========================================

class DatasetScanner:
    """
    Scans a physical dataset directory and builds a manifest.
    """

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def scan_files(self):

        records = []

        for file in self.dataset_path.rglob("*"):

            if file.is_file():

                relative = file.relative_to(
                    self.dataset_path
                )

                records.append(
                    {
                        "file_name": file.name,
                        "absolute_path": str(file),
                        "relative_path": str(relative).replace(
                            "\\",
                            "/"
                        ),
                        "extension": file.suffix.lower(),
                        "size_bytes": file.stat().st_size,
                    }
                )

        return pd.DataFrame(records)

    def detect_species(self, df):

        sources = []
        species = []

        for path in df["relative_path"]:

            parts = Path(path).parts

            if len(parts) >= 3:

                sources.append(parts[0])
                species.append(parts[1])

            elif len(parts) == 2:

                sources.append("unknown")
                species.append(parts[0])

            else:

                sources.append("unknown")
                species.append("unknown")

        df["source"] = sources
        df["species"] = species

        return df


# ==========================================
# BUILD OR LOAD MANIFEST
# ==========================================

if USE_EXISTING_MANIFEST:

    print(
        f"Loading existing manifest: "
        f"{EXISTING_MANIFEST_PATH}"
    )

    if not EXISTING_MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Existing manifest not found:\n"
            f"{EXISTING_MANIFEST_PATH}\n\n"
            f"Make sure dataset_manifest.csv is in the "
            f"same folder as this script."
        )

    manifest = pd.read_csv(
        EXISTING_MANIFEST_PATH
    )

    required_cols = {
        "file_name",
        "absolute_path",
        "relative_path",
        "species",
    }

    missing = required_cols - set(
        manifest.columns
    )

    if missing:

        raise ValueError(
            f"Existing manifest is missing required "
            f"columns: {missing}"
        )

else:

    scanner = DatasetScanner(
        DATASET_PATH
    )

    manifest = scanner.scan_files()

    manifest = scanner.detect_species(
        manifest
    )


# ==========================================
# NORMALISE SPECIES LABELS
# ==========================================

manifest["species_label"] = (
    manifest["species"]
    .apply(normalise_label)
)

print(
    "Files found:",
    len(manifest)
)


# ==========================================
# 1. EXACT DUPLICATE FILE PATHS
# ==========================================

def find_duplicate_paths(df):
    """
    Flags rows where the same relative_path appears
    more than once in the manifest.

    A clean dataset should normally contain zero
    duplicate relative paths.
    """

    dup_mask = df.duplicated(
        subset=["relative_path"],
        keep=False
    )

    dup_paths = (
        df[dup_mask]
        .sort_values("relative_path")
        .copy()
    )

    dup_paths["issue_type"] = (
        "duplicate_file_path"
    )

    return dup_paths


duplicate_paths = find_duplicate_paths(
    manifest
)

print(
    "Duplicate file paths found:",
    len(duplicate_paths)
)


# ==========================================
# 2. DUPLICATE METADATA RECORDS
# ==========================================

def find_duplicate_records(df, subset=None):
    """
    Flags potential duplicate metadata records.

    The default metadata fingerprint is:

        species_label
        source
        size_bytes

    Records with unknown species/source are excluded
    because identical file size alone is not sufficient
    evidence of a duplicate metadata record.
    """

    if subset is None:

        subset = [
            "species_label",
            "source",
            "size_bytes"
        ]

    subset = [
        column
        for column in subset
        if column in df.columns
    ]

    valid = df.copy()

    # Remove records where species information is unknown
    if "species_label" in valid.columns:

        valid = valid[
            ~valid["species_label"]
            .astype(str)
            .str.lower()
            .str.strip()
            .isin(
                [
                    "",
                    "unknown",
                    "nan",
                    "none"
                ]
            )
        ]

    # Remove records where source information is unknown
    if "source" in valid.columns:

        valid = valid[
            ~valid["source"]
            .astype(str)
            .str.lower()
            .str.strip()
            .isin(
                [
                    "",
                    "unknown",
                    "nan",
                    "none"
                ]
            )
        ]

    dup_mask = valid.duplicated(
        subset=subset,
        keep=False
    )

    dup_records = (
        valid[dup_mask]
        .sort_values(subset)
        .copy()
    )

    dup_records["issue_type"] = (
        "duplicate_metadata_record"
    )

    return dup_records


duplicate_records = find_duplicate_records(
    manifest
)

print(
    "Duplicate metadata records found:",
    len(duplicate_records)
)


# ==========================================
# 3. FILE-HASH BASED AUDIO DUPLICATE DETECTION
# ==========================================

def hash_file(
    path,
    algorithm=HASH_ALGORITHM,
    chunk_size=HASH_CHUNK_SIZE
):
    """
    Creates a SHA-256 hash of a file.

    The file is read in chunks so large audio files
    do not need to be loaded into memory at once.

    If the physical file does not exist, the function
    returns FILE_NOT_FOUND.
    """

    path = Path(path)

    # Check whether the path exists
    if not path.exists():

        return "ERROR:FILE_NOT_FOUND"

    # Check whether the path is actually a file
    if not path.is_file():

        return "ERROR:NOT_A_FILE"

    hasher = hashlib.new(
        algorithm
    )

    try:

        with open(path, "rb") as f:

            for chunk in iter(
                lambda: f.read(chunk_size),
                b""
            ):

                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception as e:

        return f"ERROR:{e}"


def find_duplicate_hashes(df):
    """
    Detects byte-for-byte duplicate audio files
    using SHA-256.

    If the physical audio files are unavailable,
    records are placed into hash_errors rather than
    incorrectly reporting that no duplicates exist.
    """

    tqdm.pandas(
        desc="Hashing audio files"
    )

    df = df.copy()

    print(
        "\nStarting SHA-256 audio duplicate check..."
    )

    df["file_hash"] = (
        df["absolute_path"]
        .progress_apply(hash_file)
    )

    # Successfully hashed files
    hash_ok = df[
        ~df["file_hash"]
        .astype(str)
        .str.startswith("ERROR:")
    ].copy()

    # Find duplicate hashes
    dup_mask = hash_ok.duplicated(
        subset=["file_hash"],
        keep=False
    )

    dup_hashes = (
        hash_ok[dup_mask]
        .sort_values("file_hash")
        .copy()
    )

    dup_hashes["issue_type"] = (
        "duplicate_audio_content"
    )

    # Files that could not be read
    hash_errors = df[
        df["file_hash"]
        .astype(str)
        .str.startswith("ERROR:")
    ].copy()

    hash_errors["issue_type"] = (
        "hash_read_error"
    )

    return dup_hashes, hash_errors


# IMPORTANT:
# Actually run the hash duplicate check.

duplicate_hashes, hash_errors = (
    find_duplicate_hashes(manifest)
)

print(
    "Duplicate audio files (by hash) found:",
    len(duplicate_hashes)
)

if len(hash_errors):

    print(
        "Files that could not be hashed:",
        len(hash_errors)
    )


# ==========================================
# 4. SPECIES-NAME FORMATTING /
#    INCONSISTENT IDENTIFIERS
# ==========================================

def find_species_inconsistencies(df):
    """
    Performs two related checks.

    A. Species formatting issue:
       The raw species name differs from the normalised
       species label.

    B. Inconsistent species identifier:
       Multiple raw species names map to the same
       normalised species label.

    Example:

        Eastern-Rosella
        eastern rosella

    Both become:

        eastern_rosella
    """

    results = []

    # ------------------------------------------
    # A. Formatting differences
    # ------------------------------------------

    formatting_issues = df[
        df["species"] != df["species_label"]
    ].copy()

    formatting_issues["issue_type"] = (
        "species_name_formatting"
    )

    results.append(
        formatting_issues
    )

    # ------------------------------------------
    # B. Multiple raw variants
    # ------------------------------------------

    variants_per_label = (
        df.groupby("species_label")["species"]
        .nunique()
    )

    inconsistent_labels = (
        variants_per_label[
            variants_per_label > 1
        ]
        .index
        .tolist()
    )

    if inconsistent_labels:

        colliding = df[
            df["species_label"]
            .isin(inconsistent_labels)
        ].copy()

        colliding["issue_type"] = (
            "inconsistent_species_identifier"
        )

        results.append(
            colliding
        )

    # ------------------------------------------
    # Combine results
    # ------------------------------------------

    if results:

        return (
            pd.concat(
                results,
                ignore_index=True
            )
            .drop_duplicates(
                subset=[
                    "absolute_path",
                    "issue_type"
                ]
            )
        )

    return pd.DataFrame(
        columns=list(df.columns)
        + ["issue_type"]
    )


species_inconsistencies = (
    find_species_inconsistencies(
        manifest
    )
)

print(
    "Species naming inconsistencies found:",
    len(species_inconsistencies)
)


# ==========================================
# 5. MISSING SOURCE / LOCATION / LABEL
# ==========================================

def find_missing_metadata(df):
    """
    Flags records missing critical metadata.

    Checks:
        - source
        - species
        - species_label
        - location (if available)

    The manifest currently does not appear to contain
    a location column, so location cannot be assessed
    unless that field is present.
    """

    missing_markers = [
        "",
        "unknown",
        "nan",
        "none"
    ]

    check_columns = [
        column
        for column in [
            "source",
            "species",
            "species_label",
            "location"
        ]
        if column in df.columns
    ]

    frames = []

    for col in check_columns:

        mask = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
            .isin(missing_markers)
        )

        subset = df[mask].copy()

        if len(subset):

            subset["issue_type"] = (
                f"missing_{col}"
            )

            frames.append(
                subset
            )

    if frames:

        return pd.concat(
            frames,
            ignore_index=True
        )

    return pd.DataFrame(
        columns=list(df.columns)
        + ["issue_type"]
    )


missing_metadata = find_missing_metadata(
    manifest
)

print(
    "Records with missing metadata found:",
    len(missing_metadata)
)


# ==========================================
# LOCATION AVAILABILITY MESSAGE
# ==========================================

if "location" not in manifest.columns:

    print(
        "Location metadata check: NOT AVAILABLE "
        "- manifest does not contain a 'location' column."
    )


# ==========================================
# 6. COMBINED STRUCTURED ISSUE LIST
# ==========================================

def build_issue_list(*frames):
    """
    Merges all checks into one structured
    issue-tracker-style table.

    Main fields:

        file_name
        relative_path
        species
        species_label
        source
        issue_type
        severity
    """

    severity_map = {

        "duplicate_file_path":
            "High",

        "duplicate_metadata_record":
            "Medium",

        "duplicate_audio_content":
            "High",

        "hash_read_error":
            "Medium",

        "species_name_formatting":
            "Low",

        "inconsistent_species_identifier":
            "Medium",

        "missing_source":
            "High",

        "missing_species":
            "High",

        "missing_species_label":
            "High",

        "missing_location":
            "Medium",
    }

    keep_cols = [
        "file_name",
        "relative_path",
        "species",
        "species_label",
        "source",
        "issue_type"
    ]

    combined = []

    for frame in frames:

        if frame is None:
            continue

        if len(frame) == 0:
            continue

        cols = [
            column
            for column in keep_cols
            if column in frame.columns
        ]

        if cols:

            combined.append(
                frame[cols]
            )

    if not combined:

        return pd.DataFrame(
            columns=keep_cols
            + ["severity"]
        )

    issues = pd.concat(
        combined,
        ignore_index=True
    )

    issues["severity"] = (
        issues["issue_type"]
        .map(severity_map)
        .fillna("Low")
    )

    # Sort High -> Medium -> Low
    severity_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2
    }

    issues["_severity_order"] = (
        issues["severity"]
        .map(severity_order)
        .fillna(3)
    )

    issues = (
        issues
        .sort_values(
            [
                "_severity_order",
                "issue_type"
            ]
        )
        .drop(
            columns=["_severity_order"]
        )
    )

    return issues


# Build the final combined issue list

issue_list = build_issue_list(

    duplicate_paths,

    duplicate_records,

    duplicate_hashes,

    hash_errors,

    species_inconsistencies,

    missing_metadata,

)


# ==========================================
# SAVE REPORTS
# ==========================================

duplicate_paths.to_csv(
    REPORT_OUTPUT / "duplicate_paths.csv",
    index=False
)

duplicate_records.to_csv(
    REPORT_OUTPUT / "duplicate_records.csv",
    index=False
)

duplicate_hashes.to_csv(
    REPORT_OUTPUT / "duplicate_hashes.csv",
    index=False
)

hash_errors.to_csv(
    REPORT_OUTPUT / "hash_errors.csv",
    index=False
)

species_inconsistencies.to_csv(
    REPORT_OUTPUT
    / "metadata_inconsistencies_species.csv",
    index=False
)

missing_metadata.to_csv(
    REPORT_OUTPUT
    / "metadata_inconsistencies_missing.csv",
    index=False
)

issue_list.to_csv(
    REPORT_OUTPUT
    / "dataset_issues_report.csv",
    index=False
)


# ==========================================
# FINAL SUMMARY
# ==========================================

print(
    "\n=========================================="
)

print(
    "DATASET QA COMPLETE"
)

print(
    "=========================================="
)

print(
    f"Total manifest records: "
    f"{len(manifest)}"
)

print(
    f"Duplicate file paths: "
    f"{len(duplicate_paths)}"
)

print(
    f"Duplicate metadata records: "
    f"{len(duplicate_records)}"
)

print(
    f"Duplicate audio content: "
    f"{len(duplicate_hashes)}"
)

print(
    f"Hash/read errors: "
    f"{len(hash_errors)}"
)

print(
    f"Species naming inconsistencies: "
    f"{len(species_inconsistencies)}"
)

print(
    f"Missing metadata records: "
    f"{len(missing_metadata)}"
)

print(
    f"Total structured issues: "
    f"{len(issue_list)}"
)

print(
    "\nReports saved to:"
)

print(
    REPORT_OUTPUT
)

print(
    "\nGenerated files:"
)

print(
    " - duplicate_paths.csv"
)

print(
    " - duplicate_records.csv"
)

print(
    " - duplicate_hashes.csv"
)

print(
    " - hash_errors.csv"
)

print(
    " - metadata_inconsistencies_species.csv"
)

print(
    " - metadata_inconsistencies_missing.csv"
)

print(
    " - dataset_issues_report.csv"
    "  (combined structured issue list)"
)

print(
    "=========================================="
)
