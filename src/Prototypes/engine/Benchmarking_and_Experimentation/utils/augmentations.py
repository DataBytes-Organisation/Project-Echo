# augmentatiosns.py
# This file contains the augmentations used in the experiments.
# It is a separate file to keep the code clean and organized.


# disable warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*AutoGraph.*")

# environment settings
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# generic libraries
from platform import python_version
import functools
from functools import lru_cache
import diskcache as dc
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import math

# tensor flow / keras related libraries
import tensorflow as tf
import tensorflow_hub as hub
# import tensorflow_addons as tfa 
# from keras.utils import dataset_utils

# image processing related libraries
import librosa

# audio processing libraries
import audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from config.system_config import SC  # Import system settings
from config.model_configs import MODELS  # Import model configurations

# Image augmentation pipeline (updated to remove tf.contrib)
def tensorflow_image_augmentations(sample, label, variant, cache_key, cache_found):
    """
    After the melspectrogram pipeline is executed, a 2D image is created representing 
    the signal energy at frequency/time points. This image is augmented using the 
    function below. The function applies a random rotation of between -2 and 2 degrees.
    More advanced augmentations can be added here.
    
    Args:
        sample (tf.Tensor): The input image tensor.
        label (tf.Tensor): The corresponding label tensor.
        variant (str): The variant of the input data.
        cache_key (str): The cache key for the input data.
        cache_found (np.int32): Flag indicating if the cache was found.
        
    Returns:
        sample (tf.Tensor): The augmented image tensor.
        label (tf.Tensor): The corresponding label tensor.
        variant (str): The variant of the input data.
        cache_key (str): The cache key for the input data.
        cache_found (np.int32): Flag indicating if the cache was found.
        """

       
    if cache_found == np.int32(0):
        # Random rotation -2 deg to 2 deg
        degrees = tf.random.uniform(shape=(), minval=-2, maxval=2)

        # Convert the angle to radians
        radians = degrees * math.pi / 180.0

        # Define the transformation matrix for rotation
        def angles_to_projective_transforms(angle, image_height, image_width):
            """Creates a transformation matrix for the given angle."""
            cos_val = tf.cos(angle)
            sin_val = tf.sin(angle)
            transform = [
                cos_val, -sin_val, 0.0,
                sin_val, cos_val, 0.0,
                0.0, 0.0
            ]
            return tf.convert_to_tensor(transform, dtype=tf.float32)

        # Generate the transformation matrix
        rotation_matrix = angles_to_projective_transforms(radians, tf.shape(sample)[0], tf.shape(sample)[1])

        # Apply the rotation using the transformation matrix
        sample = tf.raw_ops.ImageProjectiveTransformV2(
            images=tf.expand_dims(sample, 0),  # Add batch dimension
            transforms=tf.expand_dims(rotation_matrix, 0),  # Expand dims for batch
            output_shape=tf.shape(sample)[:2],  # Ensure only height and width
            interpolation="BILINEAR"
        )
        # Remove the batch dimension
        sample = tf.squeeze(sample, 0)

    return sample, label, variant, cache_key, cache_found

# Audio augmentation pipeline
def python_audio_augmentations(sample, label, variant, cache_key, cache_found, prob=0.2):
    """ The following code applies a sequence of augmentations to the audio signal.
    A probability of applying the augmentation is used to ensure the augmentation 
    isn't applied every sample. This means there will be some samples that go 
    straight through with no augmentations and a small probability that in fact 
    all augmentations will be applied
    
    Args:
        sample (np.ndarray): The input audio signal.
        label (str): The corresponding label for the audio signal.
        variant (str): The variant of the input data.
        cache_key (str): The cache key for the input data.
        cache_found (np.int32): Flag indicating if the cache was found.
        prob (float): Probability of applying the augmentations.
        
    Returns:
        sample (np.ndarray): The augmented audio signal.
        label (str): The corresponding label for the audio signal.
        variant (str): The variant of the input data.
        cache_key (str): The cache key for the input data.
        cache_found (np.int32): Flag indicating if the cache was found."""
    
    p = prob

    if cache_found == np.int32(0):
        # See https://github.com/iver56/audiomentations for more options

        augmentations = Compose([
            # Add Gaussian noise with a random amplitude to the audio
            # This can help the model generalize to real-world scenarios where noise is present
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=prob),

            # Time-stretch the audio without changing its pitch
            # This can help the model become invariant to small changes in the speed of the audio
            TimeStretch(min_rate=0.8, max_rate=1.25, p=prob),

            # Shift the pitch of the audio within a range of semitones
            # This can help the model generalize to variations in pitch that may occur in real-world scenarios
            PitchShift(min_semitones=-4, max_semitones=4, p=prob),

            # Shift the audio in time by a random fraction
            # This can help the model become invariant to the position of important features in the audio
            Shift(min_shift=-0.5, max_shift=0.5, p=prob),
        ])
        
        # apply audio augmentation to the clip
        # note: this augmentation is NOT applied in the test and validation pipelines
        sample = augmentations(samples=sample, sample_rate=SC['AUDIO_SAMPLE_RATE'])
    
    return sample, label, variant, cache_key, cache_found