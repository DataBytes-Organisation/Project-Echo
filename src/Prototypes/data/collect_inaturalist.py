"""
collect_inaturalist.py - iNaturalist Data Collection for Project Echo

Queries the iNaturalist API v1 for Australian wildlife audio and image
recordings filtered by the project's 21 target species, research quality
grade, and Australian geographic region.

Usage:
    python collect_inaturalist.py [--output_dir OUTPUT] [--media_type {sounds,photos,both}]
                                  [--quality_grade {research,needs_id,any}]
                                  [--max_per_species N] [--species_file PATH]
                                  [--delay SECONDS]

Output structure:
    output_dir/
    ├── Alectura_lathami/
    │   ├── sounds/
    │   │   ├── 123456_0.wav
    │   │   └── ...
    │   └── photos/
    │       ├── 789012_0.jpg
    │       └── ...
    ├── Menura_novaehollandiae/
    │   └── ...
    └── metadata.csv
"""

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

API_BASE = "https://api.inaturalist.org/v1"
AUSTRALIA_PLACE_ID = 6744
PER_PAGE = 200  # iNaturalist max per page
REQUEST_TIMEOUT = 30

# Default species list matching src/Components/Engine/class_names.json
DEFAULT_SPECIES = [
    "Alectura lathami",
    "Menura novaehollandiae",
    "Felis catus",
    "Anas gracilis",
    "Caprimulgus macrurus",
    "Chrysococcyx minutillus",
    "Leipoa ocellata",
    "Megapodius reinwardt",
    "Eudynamys orientalis",
    "Apus pacificus",
    "Geopelia placida",
    "Centropus phasianinus",
    "Rattus norvegicus",
    "Eurostopodus argus",
    "Sus scrofa",
    "Dasyurus maculatus",
    "Uperoleia laevigata",
    "Vanellus miles",
    "Capra hircus",
    "Phasianus colchicus",
    "Canis lupus dingo",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_taxon_id(species_name: str, session: requests.Session) -> int | None:
    """Look up the iNaturalist taxon ID for a scientific name."""
    resp = session.get(
        f"{API_BASE}/taxa",
        params={"q": species_name, "rank": "species", "per_page": 5},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    for taxon in results:
        if taxon.get("name", "").lower() == species_name.lower():
            return taxon["id"]

    # Fallback: return first result if available
    if results:
        logger.warning(
            "Exact match not found for '%s', using closest: '%s' (id=%d)",
            species_name,
            results[0].get("name"),
            results[0]["id"],
        )
        return results[0]["id"]

    return None


def fetch_observations(
    taxon_id: int,
    media_type: str,
    quality_grade: str,
    session: requests.Session,
    max_results: int,
) -> list[dict]:
    """Fetch observations for a taxon from iNaturalist, handling pagination."""
    observations = []
    page = 1

    while len(observations) < max_results:
        params = {
            "taxon_id": taxon_id,
            "place_id": AUSTRALIA_PLACE_ID,
            "per_page": min(PER_PAGE, max_results - len(observations)),
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }

        if quality_grade != "any":
            params["quality_grade"] = quality_grade

        if media_type == "sounds":
            params["sounds"] = "true"
        elif media_type == "photos":
            params["photos"] = "true"
        else:
            # For "both", we do two passes externally; here just get all
            pass

        resp = session.get(
            f"{API_BASE}/observations",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        if not results:
            break

        observations.extend(results)
        total = data.get("total_results", 0)
        logger.info(
            "  Page %d: fetched %d observations (total available: %d)",
            page,
            len(results),
            total,
        )

        if page * PER_PAGE >= total:
            break
        page += 1

    return observations[:max_results]


def download_file(
    url: str, dest_path: Path, session: requests.Session
) -> bool:
    """Download a file from URL to dest_path. Returns True on success."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", url, e)
        return False


def get_file_extension(url: str, default: str = "") -> str:
    """Extract file extension from URL path."""
    parsed = urlparse(url)
    path = parsed.path
    ext = os.path.splitext(path)[1]
    return ext if ext else default


def process_species(
    species_name: str,
    taxon_id: int,
    media_type: str,
    quality_grade: str,
    output_dir: Path,
    session: requests.Session,
    max_per_species: int,
    delay: float,
) -> list[dict]:
    """Process a single species: fetch observations and download media."""
    safe_name = species_name.replace(" ", "_")
    species_dir = output_dir / safe_name
    metadata_rows = []

    media_types_to_fetch = (
        ["sounds", "photos"] if media_type == "both" else [media_type]
    )

    for mtype in media_types_to_fetch:
        logger.info(
            "Fetching %s observations for %s (taxon_id=%d)...",
            mtype,
            species_name,
            taxon_id,
        )
        observations = fetch_observations(
            taxon_id, mtype, quality_grade, session, max_per_species
        )
        logger.info(
            "Found %d observations with %s for %s",
            len(observations),
            mtype,
            species_name,
        )

        download_count = 0
        for obs in observations:
            obs_id = obs["id"]

            if mtype == "sounds":
                sounds = obs.get("sounds", [])
                for i, sound in enumerate(sounds):
                    file_url = sound.get("file_url")
                    if not file_url:
                        continue
                    ext = get_file_extension(file_url, ".wav")
                    dest = species_dir / "sounds" / f"{obs_id}_{i}{ext}"
                    if dest.exists():
                        continue
                    if download_file(file_url, dest, session):
                        download_count += 1
                        metadata_rows.append({
                            "species": species_name,
                            "taxon_id": taxon_id,
                            "observation_id": obs_id,
                            "media_type": "sound",
                            "media_index": i,
                            "file_path": str(dest.relative_to(output_dir)),
                            "source_url": file_url,
                            "quality_grade": obs.get("quality_grade", ""),
                            "latitude": obs.get("geojson", {}).get("coordinates", [None, None])[1] if obs.get("geojson") else None,
                            "longitude": obs.get("geojson", {}).get("coordinates", [None, None])[0] if obs.get("geojson") else None,
                            "observed_on": obs.get("observed_on", ""),
                            "license": sound.get("license_code", ""),
                            "attribution": sound.get("attribution", ""),
                        })
                    time.sleep(delay)

            elif mtype == "photos":
                photos = obs.get("photos", [])
                for i, photo in enumerate(photos):
                    # Use original size URL
                    url = photo.get("url", "")
                    if not url:
                        continue
                    # iNaturalist photo URLs use "square" by default;
                    # replace with "original" for full resolution
                    original_url = url.replace("/square.", "/original.")
                    ext = get_file_extension(original_url, ".jpg")
                    dest = species_dir / "photos" / f"{obs_id}_{i}{ext}"
                    if dest.exists():
                        continue
                    if download_file(original_url, dest, session):
                        download_count += 1
                        metadata_rows.append({
                            "species": species_name,
                            "taxon_id": taxon_id,
                            "observation_id": obs_id,
                            "media_type": "photo",
                            "media_index": i,
                            "file_path": str(dest.relative_to(output_dir)),
                            "source_url": original_url,
                            "quality_grade": obs.get("quality_grade", ""),
                            "latitude": obs.get("geojson", {}).get("coordinates", [None, None])[1] if obs.get("geojson") else None,
                            "longitude": obs.get("geojson", {}).get("coordinates", [None, None])[0] if obs.get("geojson") else None,
                            "observed_on": obs.get("observed_on", ""),
                            "license": photo.get("license_code", ""),
                            "attribution": photo.get("attribution", ""),
                        })
                    time.sleep(delay)

        logger.info(
            "Downloaded %d %s files for %s", download_count, mtype, species_name
        )

    return metadata_rows


def main():
    parser = argparse.ArgumentParser(
        description="Collect Australian wildlife recordings from iNaturalist"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inaturalist_data",
        help="Root directory for downloaded data (default: inaturalist_data)",
    )
    parser.add_argument(
        "--media_type",
        choices=["sounds", "photos", "both"],
        default="both",
        help="Type of media to download (default: both)",
    )
    parser.add_argument(
        "--quality_grade",
        choices=["research", "needs_id", "any"],
        default="research",
        help="iNaturalist quality grade filter (default: research)",
    )
    parser.add_argument(
        "--max_per_species",
        type=int,
        default=500,
        help="Maximum observations to fetch per species per media type (default: 500)",
    )
    parser.add_argument(
        "--species_file",
        type=str,
        default=None,
        help="Path to JSON file with species list (default: built-in 21 species)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between downloads to respect rate limits (default: 1.0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load species list
    if args.species_file:
        with open(args.species_file) as f:
            species_list = json.load(f)
        logger.info("Loaded %d species from %s", len(species_list), args.species_file)
    else:
        species_list = DEFAULT_SPECIES
        logger.info("Using default %d target species", len(species_list))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "ProjectEcho-DataCollection/1.0 (Deakin University DataBytes)"
    })

    # Resolve taxon IDs
    logger.info("Resolving taxon IDs...")
    taxon_map: dict[str, int] = {}
    for species in species_list:
        tid = resolve_taxon_id(species, session)
        if tid:
            taxon_map[species] = tid
            logger.info("  %s -> taxon_id %d", species, tid)
        else:
            logger.error("  Could not resolve taxon ID for '%s', skipping", species)
        time.sleep(args.delay)

    logger.info(
        "Resolved %d / %d species", len(taxon_map), len(species_list)
    )

    # Collect data
    all_metadata: list[dict] = []
    for species, taxon_id in taxon_map.items():
        logger.info("=" * 60)
        logger.info("Processing: %s (taxon_id=%d)", species, taxon_id)
        logger.info("=" * 60)
        rows = process_species(
            species_name=species,
            taxon_id=taxon_id,
            media_type=args.media_type,
            quality_grade=args.quality_grade,
            output_dir=output_dir,
            session=session,
            max_per_species=args.max_per_species,
            delay=args.delay,
        )
        all_metadata.extend(rows)

    # Write metadata CSV
    if all_metadata:
        csv_path = output_dir / "metadata.csv"
        fieldnames = [
            "species",
            "taxon_id",
            "observation_id",
            "media_type",
            "media_index",
            "file_path",
            "source_url",
            "quality_grade",
            "latitude",
            "longitude",
            "observed_on",
            "license",
            "attribution",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)
        logger.info("Metadata written to %s (%d records)", csv_path, len(all_metadata))
    else:
        logger.warning("No data was collected.")

    logger.info("Collection complete.")


if __name__ == "__main__":
    main()
