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
    augmentation_type: str = "none"
):
    """
    Apply image/spectrogram augmentations dynamically based on IMAGE_AUGMENTATIONS.

    Args:
      sample (tf.Tensor): The spectrogram image tensor (H x W).
      label, variant, cache_key, cache_found: pipeline metadata.
      augmentation_type (str): One of the keys in IMAGE_AUGMENTATIONS.

    Returns:
      sample, label, variant, cache_key, cache_found
    """
    if cache_found == np.int32(0):
        for aug in IMAGE_AUGMENTATIONS.get(augmentation_type, []):
            t = aug["type"]
            if t == "Rotation":
                # Uniform angle between min and max in radians
                degrees = tf.random.uniform([], aug["min_degrees"], aug["max_degrees"])
                radians = degrees * math.pi / 180.0
                c = tf.cos(radians)
                s = tf.sin(radians)
                # TensorFlow expects a projective transform of 8 elements: [a0, a1, a2, b0, b1, b2, c0, c1]
                # For affine transforms, c0 and c1 should be 0.
                transform = tf.convert_to_tensor([c, -s, 0.0, s, c, 0.0, 0.0, 0.0], dtype=tf.float32)
                sample = tf.raw_ops.ImageProjectiveTransformV2(
                    images=tf.expand_dims(sample, 0),
                    transforms=tf.expand_dims(transform, 0),
                    output_shape=tf.shape(sample)[:2],
                    interpolation='BILINEAR'
                )
                sample = tf.squeeze(sample, 0)
            elif t == "SpecAugment":
                # Frequency and time masking
                freq_mask_param = aug.get("freq_mask_param", 0)
                time_mask_param = aug.get("time_mask_param", 0)
                n_freq_mask = aug.get("n_freq_mask", 0)
                n_time_mask = aug.get("n_time_mask", 0)
                # Apply frequency masks
                for _ in range(n_freq_mask):
                    f = tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32)
                    f0 = tf.random.uniform([], 0, tf.shape(sample)[0] - f, dtype=tf.int32)
                    upper = f0 + f
                    mask = tf.concat([
                        tf.ones([f0, tf.shape(sample)[1]], dtype=sample.dtype),
                        tf.zeros([f, tf.shape(sample)[1]], dtype=sample.dtype),
                        tf.ones([tf.shape(sample)[0] - upper, tf.shape(sample)[1]], dtype=sample.dtype),
                    ], axis=0)
                    sample = sample * mask
                # Apply time masks
                for _ in range(n_time_mask):
                    t_mask = tf.random.uniform([], 0, time_mask_param, dtype=tf.int32)
                    t0 = tf.random.uniform([], 0, tf.shape(sample)[1] - t_mask, dtype=tf.int32)
                    end = t0 + t_mask
                    mask = tf.concat([
                        tf.ones([tf.shape(sample)[0], t0], dtype=sample.dtype),
                        tf.zeros([tf.shape(sample)[0], t_mask], dtype=sample.dtype),
                        tf.ones([tf.shape(sample)[0], tf.shape(sample)[1] - end], dtype=sample.dtype),
                    ], axis=1)
                    sample = sample * mask
    return sample, label, variant, cache_key, cache_found
