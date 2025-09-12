# Configuration parameters (update paths as needed)
SC = {
    'AUDIO_DATA_DIRECTORY': r"C:\Users\tianc\Downloads\echo_data",  
    'CACHE_DIRECTORY':      r"C:\Users\tianc\Downloads\echo_cache",  
    'OUTPUT_DIRECTORY':     r"C:\Users\tianc\Downloads\echo_output", 

    'AUDIO_CLIP_DURATION': 5,  # seconds
    'AUDIO_SAMPLE_RATE': 48000,
    'AUDIO_NFFT': 2048,
    'AUDIO_STRIDE': 200,
    'AUDIO_MELS': 260,
    'AUDIO_FMIN': 20,
    'AUDIO_FMAX': 13000,
    'AUDIO_TOP_DB': 80,
    'AUDIO_WINDOW': None,
    'AUDIO_AUGMENTATION': 'none',
    'IMAGE_AUGMENTATION': 'none',

    'MODEL_INPUT_IMAGE_WIDTH':  224,
    'MODEL_INPUT_IMAGE_HEIGHT': 224,
    'MODEL_INPUT_IMAGE_CHANNELS': 3,


    'CLASSIFIER_BATCH_SIZE': 16,
    'MAX_EPOCHS': 50,
    'USE_DISK_CACHE': False,
    'SAMPLE_VARIANTS': 20,
}