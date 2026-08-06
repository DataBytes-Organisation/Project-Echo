from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import re

PROJECT_ROOT = Path.cwd()

#change the folder names to the required files
DATASET_ZIP_NAME = "datasets.zip"
DATASET_FOLDER_NAME = "datasets"

ZIP_PATH = PROJECT_ROOT / DATASET_ZIP_NAME
DATASET_FOLDER = PROJECT_ROOT / DATASET_FOLDER_NAME

REPORT_OUTPUT = PROJECT_ROOT / "reports"

REPORT_OUTPUT.mkdir(
    exist_ok=True
)

SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
]

MIN_DURATION = 1.0
MAX_DURATION = 10.0

#class
class DatasetScanner:

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)


    def scan_files(self):
        """
        Scan every file in dataset.
        """

        records = []

        for file in self.dataset_path.rglob("*"):

            if file.is_file():

                relative = file.relative_to(self.dataset_path)

                records.append({

                    "file_name": file.name,

                    "absolute_path": str(file),

                    "relative_path": str(relative).replace("\\",
"/"),

                    "extension": file.suffix.lower(),

                    "size_bytes": file.stat().st_size

                })


        return pd.DataFrame(records)

    def check_paths(self, df):

        """
    Check for inconsistent folder structures.
    Expected format:
        dataset/source/species/audio_file
    Also checks whether all files have a consistent folder depth.
    """

        depths = []
        status = []

      # Count folder depth for every file
        for path in df["relative_path"]:
            depths.append(len(Path(path).parts))

      # Most common folder depth
        expected_depth = max(set(depths), key=depths.count)

      # Validate each path
        for path, depth in zip(df["relative_path"], depths):
            parts = Path(path).parts

            if len(parts) < 3:
                status.append("Missing source or species folder")

            elif parts[0] == "":
                status.append("Invalid species folder")

            elif depth != expected_depth:
                status.append(f"Inconsistent depth ({depth} levels)")

            else:
                status.append("OK")
        df["path_status"] = status

        return df

    def detect_species(self, df):
        """
        Expected structure:

        dataset/
            source/
                species/
                    audio_file

        Example:

        datasets/
            gcp/
                bird_001/
                    audio.wav

        Output:
            source = gcp
            species = bird_001
        """

        sources = []
        species = []

        for path in df["relative_path"]:

            parts = Path(path).parts

            if len(parts) >= 3:
                # source/species/file
                sources.append(parts[0])
                species.append(parts[1])

            elif len(parts) == 2:
                # species/file (fallback)
                sources.append("unknown")
                species.append(parts[0])

            else:
                sources.append("unknown")
                species.append("unknown")


        df["source"] = sources
        df["species"] = species

        return df



scanner = DatasetScanner(DATASET_PATH)


manifest = scanner.scan_files()

manifest = scanner.detect_species(manifest)


print("Files found:",len(manifest))


#helper functions 
# ==========================================
# LABEL NORMALISATION
# ==========================================
def normalise_label(name):

    name=str(name).lower().strip()

    name=re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    name=re.sub(
        r"_+",
        "_",
        name
    )

    return name.strip("_")



manifest["species_label"] = (
    manifest["species"]
    .apply(normalise_label)
)


# ==========================================
# STRUCTURE VALIDATION
# ==========================================
def validate_structure(df):

    results=[]


    # Empty files

    empty_files=df[
        df["size_bytes"]==0
    ]


    results.append({

        "check":
        "Empty files",

        "count":
        len(empty_files)

    })


    # Missing labels

    missing_labels=df[
        df["species_label"]
        .isin(["unknown",""])
    ]


    results.append({

        "check":
        "Missing species labels",

        "count":
        len(missing_labels)

    })


    # Unsupported files

    unsupported=df[
        ~df.extension.isin(
            SUPPORTED_AUDIO_EXTENSIONS
        )
    ]


    results.append({

        "check":
        "Unsupported extensions",

        "count":
        len(unsupported)

    })


    return pd.DataFrame(results)



structure_report = validate_structure(manifest)


#add the unsupported extension to the manifest
def validate_format(extension):

    if extension in SUPPORTED_AUDIO_EXTENSIONS:
        return "OK"

    else:
        return "unsupported_format"

manifest["format_status"] = (
    manifest["extension"]
    .apply(validate_format)
)

# ==========================================
# AUDIO VALIDATION
# ==========================================


def validate_audio(path):

    result={

        "readable":False,

        "duration":None,

        "sample_rate":None,

        "channels":None,

        "issue":""

    }


    try:

        audio,sr = librosa.load(
            path,
            sr=None,
            mono=False
        )


        result["readable"]=True

        result["sample_rate"]=sr


        if audio.ndim==1:
            channels=1

        else:
            channels=audio.shape[0]


        result["channels"]=channels


        duration=librosa.get_duration(
            y=audio,
            sr=sr
        )


        result["duration"]=round(
            duration,
            3
        )


        if duration < MIN_DURATION:

            result["issue"]="too_short"


        elif duration > MAX_DURATION:

            result["issue"]="too_long"


    except Exception as e:

        result["issue"]="unreadable"



    return result

#add this analysis too as a addition - acoustic complexity with different sounds
def analyse_acoustic_complexity(
    file_path,
    sample_rate=16000
):
    """
    Analyse environmental audio complexity.
    Only generates quality indicators.
    """

    result = {
        "file_path": file_path,
        "silence_ratio": None,
        "mean_energy": None,
        "energy_variance": None,
        "spectral_centroid": None,
        "quality_flag": None,
        "error": None
    }


    try:

        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=True
        )


        # RMS energy
        rms = librosa.feature.rms(
            y=audio
        )[0]


        silence_ratio = np.mean(
            rms < 0.01
        )


        mean_energy = np.mean(rms)

        energy_variance = np.var(rms)


        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(
                y=audio,
                sr=sample_rate
            )
        )


        # Quality rules
        if silence_ratio > 0.8:

            flag = "HIGH_SILENCE"


        elif mean_energy < 0.01:

            flag = "LOW_SIGNAL"


        elif energy_variance > 0.05:

            flag = "HIGH_VARIABILITY"


        else:

            flag = "GOOD"


        result.update({

            "silence_ratio":
                round(float(silence_ratio),4),

            "mean_energy":
                round(float(mean_energy),6),

            "energy_variance":
                round(float(energy_variance),6),

            "spectral_centroid":
                round(float(spectral_centroid),3),

            "quality_flag":
                flag
        })


    except Exception as e:
        # Record audio analysis errors without stopping the pipeline.
        result["quality_flag"] = "ERROR"
        result["error"] = str(e)


    return result

#main execution 
if __name__ == "__main__":

    print("=" * 60)
    print("Dataset QA Pipeline Started")
    print("=" * 60)


    scanner = DatasetScanner(
        DATASET_FOLDER
    )


    # Dataset discovery
    manifest = scanner.scan_files()


    # Extract source and species labels
    manifest = scanner.detect_species(
        manifest
    )


    # Normalise labels
    manifest["species_label"] = (
        manifest["species"]
        .apply(normalise_label)
    )


    # Validate paths
    manifest = scanner.check_paths(
        manifest
    )


    # Validate formats
    manifest["format_status"] = (
        manifest["extension"]
        .apply(validate_format)
    )


    # Structure report
    structure_report = validate_structure(
        manifest
    )


    # Audio validation
    audio_results = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest)
    ):

        audio_results.append(
            validate_audio(
                row["absolute_path"]
            )
        )


    audio_df = pd.DataFrame(
        audio_results
    )


    manifest = pd.concat(
        [
            manifest.reset_index(drop=True),
            audio_df.reset_index(drop=True)
        ],
        axis=1
    )


    # Acoustic analysis
    acoustic_results = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest)
    ):

        result = analyse_acoustic_complexity(
            row["absolute_path"]
        )

        result["species"] = row["species_label"]
        result["source"] = row["source"]

        acoustic_results.append(result)


    acoustic_report = pd.DataFrame(
        acoustic_results
    )


    # Counts
    species_counts = (
        manifest
        .groupby("species_label")
        .size()
        .reset_index(name="file_count")
    )


    source_species_counts = (
        manifest
        .groupby(
            [
                "source",
                "species_label"
            ]
        )
        .size()
        .reset_index(name="file_count")
    )


    # Export reports
    manifest.to_csv(
        REPORT_OUTPUT / "dataset_manifest.csv",
        index=False
    )


    structure_report.to_csv(
        REPORT_OUTPUT / "structure_report.csv",
        index=False
    )


    species_counts.to_csv(
        REPORT_OUTPUT / "species_counts.csv",
        index=False
    )


    source_species_counts.to_csv(
        REPORT_OUTPUT / "source_species_counts.csv",
        index=False
    )


    acoustic_report.to_csv(
        REPORT_OUTPUT / "acoustic_complexity_report.csv",
        index=False
    )


    print("Pipeline completed.")
