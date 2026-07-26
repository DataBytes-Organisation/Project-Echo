"""
augmentations.py

Defines the dynamic augmentation pipeline functions by importing settings
from augmentation_configs.py. Maintains existing function signatures and
mirrors the original hard-coded augmentations via configuration keys.
"""

import numpy as np
import tensorflow as tf
import math
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from config.system_config import SC
from config.augmentation_configs import AUDIO_AUGMENTATIONS, IMAGE_AUGMENTATIONS


def python_audio_augmentations(
    sample, label, variant, cache_key, cache_found,
    augmentation_type: str = "basic"
):
    """
    Apply audio augmentations dynamically based on augmentation_configs.

    Args:
      sample (np.ndarray): The raw audio signal.
      label, variant, cache_key, cache_found: pipeline metadata.
      augmentation_type (str): One of the keys in AUDIO_AUGMENTATIONS.

    Returns:
      sample, label, variant, cache_key, cache_found
    """
    if cache_found == np.int32(0):
        ops = []
        for aug in AUDIO_AUGMENTATIONS.get(augmentation_type, []):
            t = aug["type"]
            if t == "AddGaussianNoise":
                ops.append(AddGaussianNoise(
                    min_amplitude=aug["min_amplitude"],
                    max_amplitude=aug["max_amplitude"],
                    p=aug["p"]
                ))
            elif t == "TimeStretch":
                ops.append(TimeStretch(
                    min_rate=aug["min_rate"],
                    max_rate=aug["max_rate"],
                    p=aug["p"]
                ))
            elif t == "PitchShift":
                ops.append(PitchShift(
                    min_semitones=aug["min_semitones"],
                    max_semitones=aug["max_semitones"],
                    p=aug["p"]
                ))
            elif t == "Shift":
                ops.append(Shift(
                    min_shift=aug["min_shift"],
                    max_shift=aug["max_shift"],
                    p=aug["p"]
                ))
        if ops:
            augmenter = Compose(ops)
            sample = augmenter(samples=sample, sample_rate=SC['AUDIO_SAMPLE_RATE'])
    return sample, label, variant, cache_key, cache_found


def tensorflow_image_augmentations(
    sample, label, variant, cache_key, cache_found,
    augmentation_type: str = None
):
    """
    Apply image/spectrogram augmentations. If augmentation_type is not provided,
    use SC['IMAGE_AUGMENTATION'].

    sample: tf.Tensor with shape (H, W) or (H, W, C)
    returns: sample, label, variant, cache_key, cache_found
    """
    if cache_found != np.int32(0):
        # Data came from disk cache; skip augmentation to keep cache consistent
        return sample, label, variant, cache_key, cache_found

    # Decide which augmentation pipeline to use
    if augmentation_type is None:
        augmentation_type = SC.get('IMAGE_AUGMENTATION', 'none')

    if augmentation_type == 'none':
        return sample, label, variant, cache_key, cache_found

    # Ensure sample is float32 and has a channel dimension
    sample = tf.cast(sample, tf.float32)
    if tf.rank(sample) == 2:
        sample = tf.expand_dims(sample, -1)  # (H, W) -> (H, W, 1)

    # === Example pipelines ===
    if augmentation_type == "basic_rotation":
        # degrees -> radians
        min_deg, max_deg = -5.0, 5.0  # or pull from your IMAGE_AUGMENTATIONS config
        degrees = tf.random.uniform([], min_deg, max_deg)
        radians = degrees * (math.pi / 180.0)
        # rotate around center; bilinear; reflect edges
        sample = tf.image.rotate(sample, radians, interpolation='BILINEAR', fill_mode='reflect')

    elif augmentation_type == "specaugment":
        # Minimal, safe SpecAugment (freq/time masking) similar to your code
        # Expect sample as (H, W, C) — apply mask on the first two dims
        # Pull params from config if you have them, else sane defaults:
        freq_mask_param = 8
        time_mask_param = 16
        n_freq_mask = 1
        n_time_mask = 1

        H = tf.shape(sample)[0]
        W = tf.shape(sample)[1]

        # Frequency mask(s)
        for _ in range(n_freq_mask):
            f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, tf.maximum(H - f, 1), dtype=tf.int32)
            mask = tf.concat([
                tf.ones([f0, W, 1], dtype=sample.dtype),
                tf.zeros([f,  W, 1], dtype=sample.dtype),
                tf.ones([H - (f0 + f), W, 1], dtype=sample.dtype),
            ], axis=0)
            sample = sample * mask

        # Time mask(s)
        for _ in range(n_time_mask):
            t = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, tf.maximum(W - t, 1), dtype=tf.int32)
            mask = tf.concat([
                tf.ones([H, t0, 1], dtype=sample.dtype),
                tf.zeros([H, t,  1], dtype=sample.dtype),
                tf.ones([H, W - (t0 + t), 1], dtype=sample.dtype),
            ], axis=1)
            sample = sample * mask

    # return exactly what the pipeline expects
    return sample, label, variant, cache_key, cache_found