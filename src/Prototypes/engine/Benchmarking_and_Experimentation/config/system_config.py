# Configuration parameters (update paths as needed)
SC = {
    'AUDIO_DATA_DIRECTORY': r"D:\Echo\Audio_data",  # set your audio data directory here
    'CACHE_DIRECTORY': r"D:\Echo\Training_cache",
    'OUTPUT_DIRECTORY': r"D:\Echo\Output",

    'AUDIO_CLIP_DURATION': 5,  # seconds
    'AUDIO_SAMPLE_RATE': 48000,
    'AUDIO_NFFT': 2048,
    'AUDIO_STRIDE': 200,
    'AUDIO_MELS': 260,
    'AUDIO_FMIN': 20,
    'AUDIO_FMAX': 13000,
    'AUDIO_TOP_DB': 80,
    'AUDIO_WINDOW': None,

    'MODEL_INPUT_IMAGE_WIDTH': 260,
    'MODEL_INPUT_IMAGE_HEIGHT': 260,
    'MODEL_INPUT_IMAGE_CHANNELS': 3,

    'CLASSIFIER_BATCH_SIZE': 16,
    'MAX_EPOCHS': 50,
    'USE_DISK_CACHE': False,
    'SAMPLE_VARIANTS': 20,
}