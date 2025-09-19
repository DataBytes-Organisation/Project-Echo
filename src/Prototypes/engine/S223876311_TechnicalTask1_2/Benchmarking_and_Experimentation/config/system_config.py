from pathlib import Path

SC = {
    # --- paths ---
    "AUDIO_DATA_DIRECTORY": "/Users/mankirat/Desktop/Deakin/DEAKIN25/Sem2/ProjectEcho/TechnicalTask1/Technical_Task2/Benchmarking_and_Experimentation/dataset",
    "CACHE_DIRECTORY": "/Users/mankirat/Desktop/Deakin/DEAKIN25/Sem2/ProjectEcho/TechnicalTask1/Technical_Task2/Benchmarking_and_Experimentation/cache",
    "OUTPUT_DIRECTORY": "/Users/mankirat/Desktop/Deakin/DEAKIN25/Sem2/ProjectEcho/TechnicalTask1/Technical_Task2/Benchmarking_and_Experimentation/results",

    # --- audio processing ---
    "AUDIO_CLIP_DURATION": 5,
    "AUDIO_SAMPLE_RATE": 48000,
    "AUDIO_NFFT": 2048,
    "AUDIO_STRIDE": 200,
    "AUDIO_MELS": 260,
    "AUDIO_FMIN": 20,
    "AUDIO_FMAX": 13000,
    "AUDIO_TOP_DB": 80,
    "AUDIO_WINDOW": None,
    "AUDIO_AUGMENTATION": "none",
    "IMAGE_AUGMENTATION": "none",

    # --- model input ---
    "MODEL_INPUT_IMAGE_WIDTH": 260,
    "MODEL_INPUT_IMAGE_HEIGHT": 260,
    "MODEL_INPUT_IMAGE_CHANNELS": 3,
    "NUM_CLASSES": 8,   # set this to however many classes you have

    # --- training ---
    "CLASSIFIER_BATCH_SIZE": 32,
    "MAX_EPOCHS": 50,
    "USE_DISK_CACHE": True,
    "SAMPLE_VARIANTS": 20,
}

# Ensure required folders exist
for key in ("AUDIO_DATA_DIRECTORY", "CACHE_DIRECTORY", "OUTPUT_DIRECTORY"):
    Path(SC[key]).mkdir(parents=True, exist_ok=True)
