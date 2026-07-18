"""
This module defines the full pipeline to create datasets for training,
validation, and testing a classification model. It includes helper functions
to wrap native Python functions for use in TensorFlow datasets, caching functions,
and the composition of pipeline steps. These functions integrate with existing
modules such as data_pipeline.py, augmentations.py, and system_config.py.
"""

import tensorflow as tf
import functools
import math
import numpy as np

# Import utility functions from other modules
from utils.data_pipeline import (
    create_datasets,
    tensorflow_add_variant_and_cache,
    python_load_and_decode_file,
    tensorflow_load_random_subsection,
    python_dataset_melspectro_pipeline,
    tensorflow_reshape_image_pipeline,
    tensorflow_drop_variant_and_cache,
    tensorflow_output_shape_setter
)
from utils.augmentations import (
    python_audio_augmentations,
    tensorflow_image_augmentations
)
from config.system_config import SC

# -----------------------------------------------------------------------------
def python_fuction_wrapper(pipeline_fn, out_types, sample, label, variant, cache_key, cache_found):
    """
    Wraps a Python function so that it can be used as part of a TensorFlow pipeline.

    This function leverages tf.numpy_function to call the provided pipeline function (pipeline_fn)
    with the arguments (sample, label, variant, cache_key, cache_found), converting the
    execution from native Python to a TensorFlow operation.

    Args:
        pipeline_fn (function): The native Python processing function to be wrapped.
        out_types (tuple): A tuple of TensorFlow dtypes for each output of pipeline_fn.
        sample: Input sample (e.g. file path or audio data).
        label: Input label.
        variant: Variant identifier for augmentations.
        cache_key: Key used for caching.
        cache_found: Flag indicating if the sample was found in cache.

    Returns:
        A tuple: (sample, label, variant, cache_key, cache_found) where each element is a tf.Tensor
        with the corresponding type defined in out_types.
    """
    sample, label, variant, cache_key, cache_found = tf.numpy_function(
        func=lambda v1, v2, v3, v4, v5: pipeline_fn(v1, v2, v3, v4, v5),
        inp=(sample, label, variant, cache_key, cache_found),
        Tout=out_types
    )
    return sample, label, variant, cache_key, cache_found

# -----------------------------------------------------------------------------
def python_disk_cache_start(sample, label, variant, cache_key, cache_found):
    """
    Initializes caching variables for the pipeline.

    Sets the cache_key to a default value and the cache_found flag to 0.
    If disk caching is enabled (per SC['USE_DISK_CACHE']), it generates a unique cache key using
    create_function_key and checks if the processed sample is already in the cache.

    Args:
        sample: The current sample.
        label: The current label.
        variant: Variant identifier.
        cache_key: Existing cache key (unused on entry).
        cache_found: Existing cache flag (unused on entry).

    Returns:
        A tuple: (sample, label, variant, cache_key, cache_found) where:
            - cache_key is set to a unique key (or default value "no key").
            - cache_found is set to 1 if the key exists in the cache, otherwise 0.
    """
    cache_key   = b'no key'
    cache_found = np.int32(0)

    if SC['USE_DISK_CACHE']:
        # create_function_key should return a tuple with a unique key for caching.
        _, cache_key, _ = create_function_key(python_disk_cache_start, sample, label, variant)
        # 'cache' must be defined as a global diskcache instance for example.
        if cache_key in cache:
            cache_found = np.int32(1)
    return sample, label, variant, cache_key, cache_found

# -----------------------------------------------------------------------------
def python_disk_cache_end(sample, label, variant, cache_key, cache_found):
    """
    Finalizes caching for the pipeline step.

    If disk caching is enabled and the processed sample was not found in the cache
    (cache_found flag is 0), then the sample is added to the cache.

    Args:
        sample: The processed sample.
        label: The sample's label.
        variant: Variant identifier.
        cache_key: Cache key (expected to be in bytes).
        cache_found: Flag indicating if the sample was already in the cache.

    Returns:
        A tuple: (sample, label, variant, cache_key, cache_found) where cache_key is decoded to string.
    """
    cache_key = cache_key.decode('utf-8')
    if SC['USE_DISK_CACHE']:
        if cache_found == np.int32(0):
            cache[cache_key] = sample  # Populate cache with the processed sample.
    return sample, label, variant, cache_key, cache_found

# -----------------------------------------------------------------------------
def create_dataset_pipeline(dataset, batch_size, is_training=True):
    """
    Constructs the full data processing pipeline from raw dataset to batched input.

    The pipeline applies a series of map operations to process the data through various stages:
      1. Shuffling (for training data)
      2. Adding a variant identifier and initializing cache variables.
      3. Starting disk caching.
      4. Loading and decoding audio files.
      5. Extracting a random audio subsection.
      6. Optionally applying audio augmentations (for training data).
      7. Generating a melspectrogram.
      8. Reshaping the output image.
      9. Optionally applying image augmentations (for training data).
      10. Ending caching.
      11. Setting output shapes.
      12. Dropping caching/variant identifiers.
      13. Batching, prefetching, and repeating.

    Args:
        dataset (tf.data.Dataset): A TensorFlow dataset containing tuples of (sample, label).
        batch_size (int): Number of samples per batch.
        is_training (bool): Flag to indicate whether to apply training augmentations.

    Returns:
        tf.data.Dataset: A batched and preprocessed dataset ready for model training or evaluation.
    """
    parallel_calls = tf.data.AUTOTUNE
    # Define expected output types for functions that use python_function_wrapper.
    cache_output_types = (tf.string, tf.float32, tf.int32, tf.string, tf.int32)
    procs_output_types = (tf.float32, tf.float32, tf.int32, tf.string, tf.int32)
    
    if is_training:
        # Apply shuffling for training datasets.
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    # Add variant information and caching initialization.
    dataset = dataset.map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
    
    # Start caching (determine if this sample is cached already).
    dataset = dataset.map(
        functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types),
        num_parallel_calls=parallel_calls
    )
    
    # Load and decode the audio file from disk.
    dataset = dataset.map(
        functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types),
        num_parallel_calls=parallel_calls
    )
    
    # Extract a random subsection of the audio.
    dataset = dataset.map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
    
    # Apply audio augmentations only for training.
    if is_training:
        dataset = dataset.map(
            functools.partial(python_fuction_wrapper, python_audio_augmentations, procs_output_types),
            num_parallel_calls=parallel_calls
        )
    
    # Generate the melspectrogram from the audio sample.
    dataset = dataset.map(
        functools.partial(python_fuction_wrapper, python_dataset_melspectro_pipeline, procs_output_types),
        num_parallel_calls=parallel_calls
    )
    
    # Reshape the image to match the expected model input dimensions.
    dataset = dataset.map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
    
    # Apply image augmentations if in training mode.
    if is_training:
        dataset = dataset.map(tensorflow_image_augmentations, num_parallel_calls=parallel_calls)
    
    # End caching: store processed sample if not already cached.
    dataset = dataset.map(
        functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types),
        num_parallel_calls=parallel_calls
    )
    
    # Cache the dataset to speed up future iterations.
    dataset = dataset.cache()
    
    # Set the shape of the output tensors.
    dataset = dataset.map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
    # Remove caching/variant identifiers from the output.
    dataset = dataset.map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
    
    # Batch, prefetch, and repeat.
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(parallel_calls)
    dataset = dataset.repeat(count=1)
    
    return dataset

# -----------------------------------------------------------------------------
def build_full_pipeline(audio_dir):
    """
    Builds the complete pipelines for training, validation, and testing based
    on the provided audio directory.

    This function first creates the raw datasets using create_datasets from
    data_pipeline.py, then applies the full processing pipeline to each split.

    Args:
        audio_dir (pathlib.Path): Path to the directory containing audio files.

    Returns:
        tuple: (train_pipeline, val_pipeline, test_pipeline, class_names) where:
            - train_pipeline: Dataset for training.
            - val_pipeline: Dataset for validation.
            - test_pipeline: Dataset for testing.
            - class_names: List of class names indexed from the audio data.
    """
    train_ds, val_ds, test_ds, class_names = create_datasets(audio_dir)
    batch_size = SC['CLASSIFIER_BATCH_SIZE']
    
    train_pipeline = create_dataset_pipeline(train_ds, batch_size, is_training=True)
    val_pipeline   = create_dataset_pipeline(val_ds, batch_size, is_training=False)
    test_pipeline  = create_dataset_pipeline(test_ds, batch_size, is_training=False)
    
    return train_pipeline, val_pipeline, test_pipeline, class_names

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    import pathlib
    audio_dir = pathlib.Path(SC['AUDIO_DATA_DIRECTORY'])
    train_pipeline, val_pipeline, test_pipeline, class_names = build_full_pipeline(audio_dir)
    
    # Iterate over a batch from the training pipeline to display output information.
    for melspectrogram, label in train_pipeline.take(1):
        print(f"Sample info: {melspectrogram.shape}")
        print(f"Label info: {label.shape}")
        # Visualize individual samples in the batch.
        import matplotlib.pyplot as plt
        for i in range(melspectrogram.shape[0]):
            plt.figure(figsize=(18, 5))
            plt.imshow(melspectrogram[i, :, :, 0].numpy().T,
                       cmap='afmhot', origin='lower', aspect='auto')
            plt.title("Label: " + str(label[i].numpy()))
            plt.show()

