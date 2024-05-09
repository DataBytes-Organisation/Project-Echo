### Set the Data Directories
SC = {
    "AUDIO_DATA_DIRECTORY": "C:\\Users\\ptanmay143\\Data\\Projects\\project-echo-data-bucket-3",
    "CACHE_DIRETORY": "C:\\Users\\ptanmay143\\Data\\Projects\\project-echo-data-cache",
    "AUDIO_CLIP_DURATION": 5,  # seconds
    "AUDIO_NFFT": 2048,
    "AUDIO_WINDOW": None,
    "AUDIO_STRIDE": 200,
    "AUDIO_SAMPLE_RATE": 48000,
    "AUDIO_MELS": 260,
    "AUDIO_FMIN": 20,
    "AUDIO_FMAX": 13000,
    "AUDIO_TOP_DB": 80,
    "MODEL_INPUT_IMAGE_WIDTH": 299,
    "MODEL_INPUT_IMAGE_HEIGHT": 299,
    "MODEL_INPUT_IMAGE_CHANNELS": 3,
    "USE_DISK_CACHE": True,
    "SAMPLE_VARIANTS": 20,
    "CLASSIFIER_BATCH_SIZE": 16,
    "MAX_EPOCHS": 5000,
}

### Set the Model Name
MODEL_NAME = "MobileNetV3Large"
