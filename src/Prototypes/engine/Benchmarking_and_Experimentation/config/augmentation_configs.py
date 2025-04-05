"""Defines different augmentation strategies to benchmark."""

# Basic audio augmentations
AUDIO_AUGMENTATIONS = {
    "none": [],
    "basic": [
        {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.5},
        {"type": "TimeStretch", "min_rate": 0.8, "max_rate": 1.25, "p": 0.5},
    ],
    "advanced": [
        {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.5},
        {"type": "TimeStretch", "min_rate": 0.8, "max_rate": 1.25, "p": 0.5},
        {"type": "PitchShift", "min_semitones": -4, "max_semitones": 4, "p": 0.5},
        {"type": "Shift", "min_fraction": -0.5, "max_fraction": 0.5, "p": 0.5},
    ],
}

# Image/spectrogram augmentations
IMAGE_AUGMENTATIONS = {
    "none": [],
    "basic_rotation": [
        {"type": "Rotation", "min_degrees": -2, "max_degrees": 2, "p": 1.0},
    ],
    "spec_augment": [
        {"type": "SpecAugment", "freq_mask_param": 27, "time_mask_param": 100, 
         "n_freq_mask": 1, "n_time_mask": 2, "p": 0.8},
    ],
    "combined": [
        {"type": "Rotation", "min_degrees": -2, "max_degrees": 2, "p": 1.0},
        {"type": "SpecAugment", "freq_mask_param": 27, "time_mask_param": 100, 
         "n_freq_mask": 1, "n_time_mask": 2, "p": 0.8},
    ],
}