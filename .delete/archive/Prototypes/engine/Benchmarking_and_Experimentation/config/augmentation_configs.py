"""Defines different augmentation strategies to benchmark."""
"""
Defines different augmentation strategies to benchmark, including programmatic grid-search
and random-search entries for audio augmentations.
Keys correspond to the 'audio_augmentation' and 'image_augmentation'
fields in experiment_configs.py.
"""

import itertools
import random

# -----------------------------------------------------------------------------
# Audio Augmentations
# -----------------------------------------------------------------------------
AUDIO_AUGMENTATIONS = {
    "none": [],

    # Single-operation strategies
    "noise_only": [
        {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.5},
    ],
    "stretch_only": [
        {"type": "TimeStretch", "min_rate": 0.8, "max_rate": 1.25, "p": 0.5},
    ],
    "pitch_only": [
        {"type": "PitchShift", "min_semitones": -4, "max_semitones": 4, "p": 0.5},
    ],
    "shift_only": [
        {"type": "Shift", "min_shift": -0.5, "max_shift": 0.5, "p": 0.5},
    ],

    # Pairwise combinations
    "noise_and_stretch": [
        {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.5},
        {"type": "TimeStretch",       "min_rate": 0.8,     "max_rate": 1.25,    "p": 0.5},
    ],
    "stretch_and_pitch": [
        {"type": "TimeStretch",       "min_rate": 0.8,     "max_rate": 1.25,    "p": 0.5},
        {"type": "PitchShift",       "min_semitones": -4,  "max_semitones": 4,   "p": 0.5},
    ],
    "noise_and_pitch": [
        {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.5},
        {"type": "PitchShift",       "min_semitones": -4,  "max_semitones": 4,   "p": 0.5},
    ],

    # All four operations
    "advanced": [
        {"type": "AddGaussianNoise", "min_amplitude": 0.001, "max_amplitude": 0.015, "p": 0.2},
        {"type": "TimeStretch",       "min_rate": 0.8,     "max_rate": 1.25,    "p": 0.2},
        {"type": "PitchShift",       "min_semitones": -4,  "max_semitones": 4,   "p": 0.2},
        {"type": "Shift",            "min_shift": -0.5,    "max_shift": 0.5,    "p": 0.2},
    ],
}

# -----------------------------------------------------------------------------
# Image Augmentations
# -----------------------------------------------------------------------------
IMAGE_AUGMENTATIONS = {
    "none": [],
    "basic_rotation": [
        {"type": "Rotation", "min_degrees": -5, "max_degrees": 5, "p": 1.0},
    ],
    "spec_augment": [
        {"type": "SpecAugment", "freq_mask_param": 30, "time_mask_param": 80,
         "n_freq_mask": 2, "n_time_mask": 2, "p": 0.8},
    ],
    "combined": [
        {"type": "Rotation",    "min_degrees": -3, "max_degrees": 3, "p": 0.8},
        {"type": "SpecAugment", "freq_mask_param": 20, "time_mask_param": 60,
         "n_freq_mask": 1,  "n_time_mask": 1,  "p": 0.7},
    ],
    "heavy": [
        {"type": "Rotation",    "min_degrees": -10, "max_degrees": 10, "p": 1.0},
        {"type": "SpecAugment", "freq_mask_param": 40,  "time_mask_param": 120,
         "n_freq_mask": 3,  "n_time_mask": 3,  "p": 1.0},
    ],
}
