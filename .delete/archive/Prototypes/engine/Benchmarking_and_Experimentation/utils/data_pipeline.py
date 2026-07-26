########################################################################################
 # These helper functions load the audio data into a 'dataset' using only paths.
 # Just dealing with paths at this early stage means the entire dataset can be shuffled in
 # memory and split before loading the actual audio data into memory.
########################################################################################

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
import hashlib
import numpy as np
import datetime
import time
import random
import io
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
from functools import partial

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
from utils.augmentations import tensorflow_image_augmentations, python_audio_augmentations   # Import augmentation configurations


from tensorflow.keras.mixed_precision import set_global_policy




def paths_and_labels_to_dataset(audio_paths, labels, num_classes):

    """Creates a TensorFlow dataset from lists of audio file paths and their corresponding labels.

    The function first converts the numerical labels into a one-hot encoded
    representation based on the total number of classes. It then creates
    separate TensorFlow datasets for the audio paths and the one-hot encoded
    labels. Finally, it zips these two datasets together, so each element
    in the resulting dataset is a pair of (audio_path, one_hot_label).

    Args:
        audio_paths (list of str): A list of strings, where each string is the
            path to an audio file.
        labels (list of int): A list of integer labels corresponding to each
            audio file in `audio_paths`.
        num_classes (int): The total number of unique classes in the dataset.
            This is used for one-hot encoding the labels.

    Returns:
        tf.data.Dataset: A TensorFlow dataset where each element is a tuple
        (audio_path, one_hot_label). `audio_path` is a tf.string tensor,
        and `one_hot_label` is a tf.float32 tensor (due to tf.one_hot).
    """
    # Convert labels to one-hot encoded format using tf.one_hot
    one_hot_labels = tf.one_hot(labels, depth=num_classes)

    # Create TensorFlow datasets for the paths and labels
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(one_hot_labels)

    # Zip both datasets into a single dataset
    zipped_path_ds = tf.data.Dataset.zip((path_ds, label_ds))
    
    return zipped_path_ds

# Function to index directories and extract class names and file paths
def index_directory(directory, file_types=('.ogg', '.mp3', '.wav', '.flac')):
    """Scans a directory structured with class-specific subdirectories to collect audio file paths and assign labels.

    This function traverses the given `directory`, assuming that each subdirectory
    represents a distinct class. It collects paths to audio files (matching
    `file_types`) within these subdirectories. Each file is assigned an integer
    label corresponding to the index of its parent subdirectory's name in a
    sorted list of all class (subdirectory) names.

    Args:
        directory (str or pathlib.Path): The path to the root directory containing
            subdirectories, where each subdirectory name is a class label.
        file_types (tuple of str, optional): A tuple of file extensions (e.g., '.wav', '.mp3')
            to be considered as audio files. Defaults to ('.ogg', '.mp3', '.wav', '.flac').

    Returns:
        tuple: A tuple containing three lists:
            - audio_files (list of str): A list of full string paths to the identified audio files.
            - labels (list of int): A list of integer labels, where each label is the
              index of the class name in `class_names` corresponding to the audio file.
            - class_names (list of str): A sorted list of unique class names,
              derived from the names of the subdirectories.
    """
    audio_files = []
    labels = []
    class_names = sorted([dir.name for dir in Path(directory).glob('*') if dir.is_dir()])
    

    for label, class_name in enumerate(class_names):
        class_dir = Path(directory) / class_name
        for file_path in class_dir.glob(f'**/*'):
            if file_path.suffix in file_types:
                audio_files.append(str(file_path))
                labels.append(label)  # Store the integer label instead of the class name

    return audio_files, labels, class_names

# Function to create the dataset
def create_datasets(audio_dir, train_split=0.7, val_split=0.2):
    """
    Create train, validation, and test datasets from audio directory.
    """
    # Convert the input string path to a Path object
    audio_dir = Path(audio_dir)

    # Check if the directory exists
    if not audio_dir.exists():
        raise ValueError(f"Directory does not exist: {audio_dir}")
    
    # Check if the directory contains audio files
    if len(list(audio_dir.glob('**/*'))) == 0:
        raise ValueError(f"No audio files found in directory: {audio_dir}")
    
    # Index the directory and get file paths and labels
    file_paths, labels, class_names = index_directory(audio_dir)

    # Convert paths and labels to a TensorFlow dataset
    dataset = paths_and_labels_to_dataset(
        audio_paths=file_paths,
        labels=labels,
        num_classes=len(class_names)
    )
    
    # Shuffle the dataset
    dataset = dataset.shuffle(len(file_paths), seed=42)

    # Calculate the size of the dataset
    dataset_size = len(file_paths)
    
    # Calculate the number of elements for each dataset split
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    return train_ds, val_ds, test_ds, class_names



def python_load_and_decode_file(sample, label, variant, cache_key, cache_found):
    """
    Loads an audio file from disk using Librosa. 
    Handles stereo to mono conversion.
    Implements disk caching to speed up loading. 
    Provides error handling with a fallback to silence (zeros).. 

    Arguments:
    sample: Path to the audio file.
    label: Label of the audio file.
    variant: Variant of the audio file.
    cache_key: Key for the cache.
    cache_found: Flag indicating if the cache was found.

    Returns:
    sample: Loaded audio data.
    label: Label of the audio file.
    variant: Variant of the audio file.
    cache_key: Key for the cache.
    cache_found: Flag indicating if the cache was found.

    """
    try:
        if cache_found == np.int32(0):
            tmp_audio_t = None
            
            with open(sample, 'rb') as file:
                # Load the audio data with librosa
                tmp_audio_t, _ = librosa.load(file, sr=SC['AUDIO_SAMPLE_RATE'])

                # Process stereo files to mono
                if tmp_audio_t.ndim == 2:
                    tmp_audio_t = tmp_audio_t[1, :]
                    
                tmp_audio_t = tmp_audio_t.astype(np.float32)
                assert(tmp_audio_t is not None)
                assert(isinstance(tmp_audio_t, np.ndarray))
                
            sample = tmp_audio_t
        else:
            sample = cache[cache_key.decode('utf-8')]

    except Exception as e:
        print(f"Error loading file {sample}: {e}")
        # Use silence (zeros) as a fallback
        sample = np.zeros(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'], dtype=np.float32)
    
    return sample, label, variant, cache_key, cache_found


def tensorflow_load_random_subsection(sample, label, variant, cache_key, cache_found):
    """
    Load a random subsection of the audio file using TensorFlow. The function loads 
    and decodes an audio file from the given path, calculates the audio file's 
    duration in seconds, and then extracts a random subsection of the specified
    duration (in seconds) from the audio. If the audio duration is shorter than
    the specified duration, the function pads the subsection with silence 
    to meet the required length. The resulting subsection is returned as a tensor.
    This function ensures that all audio samples passed down the pipeline have a consistent 
    length, which is crucial for machine learning models. The random subsection 
    extraction introduces variability into the training data, which can improve the 
    model's generalisation ability.
    
    Args:
    sample: Path to the audio file.
    label: Label of the audio file.
    variant: Variant of the audio file.
    cache_key: Key for the cache.
    cache_found: Flag indicating if the cache was found.
    
    Returns:
    sample: Loaded audio data (subsection).
    label: Label of the audio file.
    variant: Variant of the audio file.
    cache_key: Key for the cache.
    cache_found: Flag indicating if the cache was found.
    """
    
    if cache_found == np.int32(0):
        duration_secs = SC['AUDIO_CLIP_DURATION']
        
        # Determine the audio file's duration in seconds
        audio_duration_secs = tf.shape(sample)[0] / SC['AUDIO_SAMPLE_RATE']
        
        if audio_duration_secs>duration_secs:
        
            # Calculate the starting point of the 5-second subsection
            max_start = tf.cast(audio_duration_secs - duration_secs, tf.float32)
            start_time_secs = tf.random.uniform((), 0.0, max_start, dtype=tf.float32)
            
            start_index = tf.cast(start_time_secs * SC['AUDIO_SAMPLE_RATE'], dtype=tf.int32)
    
            # Load the 5-second subsection
            end_index = tf.cast(start_index + tf.cast(duration_secs, tf.int32) * SC['AUDIO_SAMPLE_RATE'], tf.int32)
            
            subsection = sample[start_index : end_index]
        
        else:
            # Pad the subsection with silence if it's shorter than 5 seconds
            padding_length = duration_secs * SC['AUDIO_SAMPLE_RATE'] - tf.shape(sample)[0]
            padding = tf.zeros([padding_length], dtype=sample.dtype)
            subsection = tf.concat([sample, padding], axis=0)

        sample = subsection

    return sample, label, variant, cache_key, cache_found

def python_dataset_melspectro_pipeline(sample, label, variant, cache_key, cache_found, model_name="EfficientNetV2B0"):

    """This function takes the audio_clip and generates a melspectrogram representation.
    The size of the image will always be the same as the length of audio is fixed and the
    various parameters such as AUDIO_STRIDE that impact image size are also fixed.
    
    Args:
        sample: Path to the audio file.
        label: Label of the audio file.
        variant: Variant of the audio file.
        cache_key: Key for the cache.
        cache_found: Flag indicating if the cache was found.
        model_name: Name of the model to determine expected input shape.
    
    Returns:
        sample: Loaded audio data (melspectrogram).
        label: Label of the audio file.
        variant: Variant of the audio file.
        cache_key: Key for the cache.
        cache_found: Flag indicating if the cache was found."""

    if cache_found == np.int32(0):
        # Get expected input shape from model config
        expected_input_shape = MODELS.get(model_name, {}).get("expected_input_shape")
        expected_height, expected_width, _ = expected_input_shape

        # Calculate the hop_length (stride) based on the desired width
        hop_length = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / expected_width)

        # Compute the mel-spectrogram
        image = librosa.feature.melspectrogram(
            y=sample, 
            sr=SC['AUDIO_SAMPLE_RATE'], 
            n_fft=SC['AUDIO_NFFT'], 
            hop_length=hop_length,  # Ensures the width is as expected
            n_mels=expected_height, # Use expected height for melspectrogram
            fmin=SC['AUDIO_FMIN'],
            fmax=SC['AUDIO_FMAX'],
            win_length=SC['AUDIO_WINDOW'])
        
        # print("Debug: melspectrogram shape:", image.shape)
        # print()
        # print(f"Hop length: {hop_length}, n_mels: {expected_height}")
        # Optionally convert the mel-spectrogram to decibel scale
        image = librosa.power_to_db(
            image, 
            top_db=SC['AUDIO_TOP_DB'], 
            ref=1.0)
        
        
        # swap axis
        image = np.moveaxis(image, 1, 0)
        sample = image
    
    return sample, label, variant, cache_key, cache_found

'''
def set_intermediate_melspec_shape(sample, label, variant, cache_key, cache_found, expected_height):
    """
    New function that sets the shape of the spectrogram tensor after tf.numpy_function.
    The height is known (n_mels/expected_height), but the width might vary slightly.
    This helps with generalisability and avoids hardcoding the width.
    """
    # sample is expected to be 2D here: (height, width)
    # Ensure sample is float32 before setting shape, just in case
    sample = tf.cast(sample, tf.float32)
    sample.set_shape([expected_height, None]) # Set known height, unknown width
    print(f"Debug: Set intermediate shape to: {sample.shape}")
    return sample, label, variant, cache_key, cache_found
'''


def tensorflow_reshape_image_pipeline(sample, label, variant, cache_key, cache_found, model_name="EfficientNetV2B0"):
    """
    This function reshapes the image to ensure it has the correct number of channels.
    It also rescales the image to the range [0, 1].
    This is important for models that expect a specific input shape and range.

    Args:
        sample: Mel-spectrogram tensor (after np.moveaxis, shape is width, height).
        label: Label of the audio file.
        variant: Variant of the audio file.
        cache_key: Key for the cache.
        cache_found: Flag indicating if the cache was found.
        model_name: Name of the model to determine expected input shape.

    Returns:
        sample: Reshaped and rescaled image tensor.
        label: Label of the audio file.
        variant: Variant of the audio file.
        cache_key: Key for the cache.
        cache_found: Flag indicating if the cache was found.
    """

    if cache_found == np.int32(0):
        # Get expected input shape from model config
        expected_input_shape = MODELS.get(model_name, {}).get("expected_input_shape")
        expected_height, expected_width, channels = expected_input_shape

        # Explicitly set the shape of the input sample (mel-spectrogram after moveaxis)
        # The height (n_mels) is known from the melspectrogram calculation.
        # The width can vary slightly before resize due to hop_length rounding.
        # After np.moveaxis in the previous step, the shape is (width, height).
        sample.set_shape([None, expected_height]) # Set partial shape (width=None, height=known)

        # reshape into standard 3 channels to add the color channel
        # Input shape: (width, height) -> Output shape: (width, height, 1)
        image = tf.expand_dims(sample, -1)

        # repeat the image to get the expected number of channels
        # Input shape: (width, height, 1) -> Output shape: (width, height, channels)
        image = tf.repeat(image, channels , axis=2)

        # note here a high quality LANCZOS5 is applied to resize the image to match model image input size
        # tf.image.resize expects target size as (height, width)
        # Input shape: (width, height, channels) -> Output shape: (expected_height, expected_width, channels)
        image = tf.image.resize(image, (expected_height, expected_width),
                                method=tf.image.ResizeMethod.LANCZOS5)

        # Ensure shape AFTER resizing to the final expected model input shape
        # The output shape of resize should be (expected_height, expected_width, channels)
        image = tf.ensure_shape(image, [expected_height, expected_width, channels])

        # rescale to range [0,1]
        image = image - tf.reduce_min(image)
        sample = image / (tf.reduce_max(image)+0.0000001) # Add epsilon to avoid division by zero

        # print("Debug: Reshaped image shape:", sample.shape)

    return sample, label, variant, cache_key, cache_found



def tensorflow_add_variant_and_cache(path, label):
    """Adds a variant ID and initializes caching variables to a data sample.

    This function generates a random variant ID, assigns the input path to the sample variable,
    and initializes the cache key and cache found flags. The variant ID is used for data augmentation,
    and the caching variables are used for disk caching.

    Args:
        path (tf.Tensor): The file path of the audio file.
        label (tf.Tensor): The one-hot encoded label for the audio file.

    Returns:
        tuple: A tuple containing:
            - sample (tf.Tensor): The file path of the audio file.
            - label (tf.Tensor): The one-hot encoded label for the audio file.
            - variant (tf.Tensor): A random integer variant ID.
            - cache_key (tf.Tensor): A default cache key (b'no key').
            - cache_found (tf.Tensor): A flag indicating whether the data is found in the cache (0).
    """
    variant     = tf.random.uniform(shape=(), minval=0, maxval=SC['SAMPLE_VARIANTS'], dtype=tf.int32)
    sample      = path
    cache_key   = b'no key'
    cache_found = np.int32(0)
    return sample, label, variant, cache_key, cache_found

def tensorflow_drop_variant_and_cache(sample, label, variant, cache_key, cache_found):
    """Removes the variant ID and caching variables from a data sample.

    This function removes the variant ID, cache key, and cache found flags from the data sample,
    returning only the sample and label. This is done after the data has been processed and cached.

    Args:
        sample (tf.Tensor): The processed audio data (mel-spectrogram).
        label (tf.Tensor): The one-hot encoded label for the audio file.
        variant (tf.Tensor): The random integer variant ID.
        cache_key (tf.Tensor): The cache key.
        cache_found (tf.Tensor): The flag indicating whether the data is found in the cache.

    Returns:
        tuple: A tuple containing:
            - sample (tf.Tensor): The processed audio data (mel-spectrogram).
            - label (tf.Tensor): The one-hot encoded label for the audio file.
    """
    return sample, label


def tensorflow_output_shape_setter(sample, label, variant, cache_key, cache_found, model_name="EfficientNetV2B0"):
    """Sets the output shape of the sample and label tensors.

    This function sets the shape of the sample tensor to [MODEL_INPUT_IMAGE_WIDTH, MODEL_INPUT_IMAGE_HEIGHT, MODEL_INPUT_IMAGE_CHANNELS]
    and the shape of the label tensor to [len(class_names), ]. This is done to ensure that the tensors have the correct shape
    for the model.

    Args:
        sample (tf.Tensor): The processed audio data (mel-spectrogram).
        label (tf.Tensor): The one-hot encoded label for the audio file.
        variant (tf.Tensor): The random integer variant ID.
        cache_key (tf.Tensor): The cache key.
        cache_found (tf.Tensor): The flag indicating whether the data is found in the cache.
        model_name (str): The name of the model (to determine expected input shape).

    Returns:
        tuple: A tuple containing:
            - sample (tf.Tensor): The processed audio data (mel-spectrogram) with the specified shape.
            - label (tf.Tensor): The one-hot encoded label for the audio file with the specified shape.
            - variant (tf.Tensor): The random integer variant ID.
            - cache_key (tf.Tensor): The cache key.
            - cache_found (tf.Tensor): The flag indicating whether the data is found in the cache.
    """

    expected_input_shape = MODELS.get(model_name, {}).get("expected_input_shape")
    expected_height, expected_width, expected_channels = expected_input_shape
    
    global class_names
    sample.set_shape([expected_height, expected_width, expected_channels]) #Changed for reusability
    label.set_shape([len(class_names),])
    return sample, label, variant, cache_key, cache_found







def python_fuction_wrapper(pipeline_fn, out_types, sample, label, variant, cache_key, cache_found):
    """Wraps a Python function for use in a TensorFlow data pipeline.

    This function takes a Python function (`pipeline_fn`) and wraps it using `tf.numpy_function`
    so that it can be used as part of a TensorFlow data pipeline. This allows you to incorporate
    arbitrary Python code into your data pipeline, such as audio decoding, feature extraction,
    or data augmentation.

    Args:
        pipeline_fn (callable): The Python function to wrap. This function should take the
            following arguments: `sample`, `label`, `variant`, `cache_key`, and `cache_found`.
        out_types (tuple): A tuple of TensorFlow data types (`tf.DType`) specifying the data
            types of the output tensors. The length of this tuple should match the number of
            values returned by `pipeline_fn`.
        sample (tf.Tensor): The input sample.
        label (tf.Tensor): The label for the input sample.
        variant (tf.Tensor): The variant ID for the input sample.
        cache_key (tf.Tensor): The cache key for the input sample.
        cache_found (tf.Tensor): A flag indicating whether the input sample was found in the cache.

    Returns:
        tuple: A tuple containing the output tensors from the wrapped Python function:
            - sample (tf.Tensor): The processed sample.
            - label (tf.Tensor): The label for the processed sample.
            - variant (tf.Tensor): The variant ID for the processed sample.
            - cache_key (tf.Tensor): The cache key for the processed sample.
            - cache_found (tf.Tensor): A flag indicating whether the processed sample was found in the cache.
    """
    # Use a lambda function to pass arguments to the function
    sample, label, variant, cache_key, cache_found = tf.numpy_function(
        func=lambda v1,v2,v3,v4,v5: pipeline_fn(v1,v2,v3,v4,v5),
        inp=(sample, label, variant, cache_key, cache_found),
        Tout=out_types)

    return sample, label, variant, cache_key, cache_found


def create_function_key(func, *args, **kwargs):
    """Generates a unique hash key for a function call based on its name, module, arguments, and keyword arguments.

    This function creates a unique key for a function call by combining the function's module, name,
    a string representation of its arguments, and a string representation of its keyword arguments.
    The key is then hashed using SHA256 to produce a shorter, consistent-length hash value. This hash
    can be used for caching function results or other purposes where a unique identifier for a function
    call is needed.

    Args:
        func (callable): The function for which to generate the key.
        *args: Positional arguments passed to the function.
        **kwargs: Keyword arguments passed to the function.

    Returns:
        tuple: A tuple containing:
            - key (str): A string representation of the function's module, name, arguments, and keyword arguments.
            - key_hash (str): A SHA256 hash of the key, providing a shorter, consistent-length identifier.
            - partial_func (functools.partial): A partial function object with the given function, arguments, and keyword arguments.
    """
    
    partial_func = functools.partial(func, *args, **kwargs)
    func_name = partial_func.func.__name__
    func_module = partial_func.func.__module__
    args_repr = repr(partial_func.args)
    kwargs_repr = repr(repr(sorted(partial_func.keywords.items())))

    key = f"{func_module}.{func_name}:{args_repr}:{kwargs_repr}"
    # Use hashlib to create a hash of the key for shorter and consistent length
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    return key, key_hash, partial_func

if SC['USE_DISK_CACHE']:
    cache = dc.Cache(SC['CACHE_DIRECTORY'], cull_limit=0, size_limit=10**9)
else:
    cache = None 

def python_disk_cache_start(sample, label, variant, cache_key, cache_found):
    """Checks if the current sample is in the disk cache and sets the cache_found flag accordingly.

    This function checks if the current sample is present in the disk cache. If it is, the cache_found
    flag is set to 1, indicating that the sample was found in the cache. If it is not, the cache_found
    flag is set to 0, indicating that the sample was not found in the cache. The cache_key is generated
    using the create_function_key function.

    Args:
        sample (tf.Tensor): The input sample.
        label (tf.Tensor): The label for the input sample.
        variant (tf.Tensor): The variant ID for the input sample.
        cache_key (tf.Tensor): The cache key for the input sample.
        cache_found (tf.Tensor): A flag indicating whether the input sample was found in the cache.

    Returns:
        tuple: A tuple containing the input tensors and the updated cache_key and cache_found tensors.
            - sample (tf.Tensor): The input sample.
            - label (tf.Tensor): The label for the input sample.
            - variant (tf.Tensor): The variant ID for the input sample.
            - cache_key (tf.Tensor): The updated cache key for the input sample.
            - cache_found (tf.Tensor): A flag indicating whether the input sample was found in the cache.
    """
    cache_key   = b'no key'
    cache_found = np.int32(0)
    
    if SC['USE_DISK_CACHE']:
        _,cache_key,_ = create_function_key(python_disk_cache_start, sample, label, variant)
        if cache_key in cache:
            #print(f'found {cache_key} in cache')
            cache_found = np.int32(1)
        else:
            pass
            #print(f'{cache_key} not found in cache')
            
    return sample, label, variant, cache_key, cache_found

def python_disk_cache_end(sample, label, variant, cache_key, cache_found):
    """Adds the current sample to the disk cache if it was not found in the cache at the start.

    This function adds the current sample to the disk cache if the cache_found flag is 0, indicating
    that the sample was not found in the cache at the start of the pipeline. If the cache_found flag is 1,
    indicating that the sample was found in the cache at the start of the pipeline, the sample is not added
    to the cache.

    Args:
        sample (tf.Tensor): The input sample.
        label (tf.Tensor): The label for the input sample.
        variant (tf.Tensor): The variant ID for the input sample.
        cache_key (tf.Tensor): The cache key for the input sample.
        cache_found (tf.Tensor): A flag indicating whether the input sample was found in the cache.

    Returns:
        tuple: A tuple containing the input tensors and the updated cache_key and cache_found tensors.
            - sample (tf.Tensor): The input sample.
            - label (tf.Tensor): The label for the input sample.
            - variant (tf.Tensor): The variant ID for the input sample.
            - cache_key (tf.Tensor): The updated cache key for the input sample.
            - cache_found (tf.Tensor): A flag indicating whether the input sample was found in the cache.
    """


    cache_key = cache_key.decode('utf-8')
    if SC['USE_DISK_CACHE']:
        # if it was not found in the cache at the start, then populate with what we built
        # during the pipeline execution
        if cache_found == np.int32(0):
            #print(f'adding {cache_key} to cache')
            cache[cache_key] = sample
        #else:
        #    sample = cache[cache_key]
            
    return sample, label, variant, cache_key, cache_found



def build_datasets(train_ds, val_ds, test_ds, class_names_list, model_name="EfficientNetV2B0"):
    """
    Creates training, validation, and test datasets with specified data pipeline transformations.

    Args:
        train_ds (tf.data.Dataset): Initial training dataset (paths and labels).
        val_ds (tf.data.Dataset): Initial validation dataset (paths and labels).
        test_ds (tf.data.Dataset): Initial test dataset (paths and labels).
        class_names_list (list): List of class names. Used to set global for shape setter.
        model_name (str, optional): Name of the model to determine expected input shape. Defaults to "EfficientNetV2B0".

    Returns:
        tuple: A tuple containing the processed training, validation, and test datasets.
    """

    print(f"Building datasets for {model_name}...")

    # Set global class_names used by tensorflow_output_shape_setter
    global class_names
    class_names = class_names_list

    # Get the length of the training dataset
    try:
        # Attempt to get cardinality, handle potential errors if dataset is infinite or unknown
        len_train_ds = tf.data.experimental.cardinality(train_ds).numpy()
        if len_train_ds == tf.data.experimental.UNKNOWN_CARDINALITY or len_train_ds == tf.data.experimental.INFINITE_CARDINALITY:
            print("Warning: Training dataset size is unknown or infinite. Shuffle buffer size set to 1000.")
            shuffle_buffer_size = 1000 # Default buffer size if cardinality is unknown/infinite
        else:
            shuffle_buffer_size = len_train_ds
    except tf.errors.InvalidArgumentError:
         print("Warning: Could not determine training dataset size. Shuffle buffer size set to 1000.")
         shuffle_buffer_size = 1000


    parallel_calls = tf.data.AUTOTUNE
    cache_output_types = (tf.string, tf.float32, tf.int32, tf.string, tf.int32)
    procs_output_types = (tf.float32, tf.float32, tf.int32, tf.string, tf.int32)

    # --- Create partial functions with model_name ---
    # For Python functions needing the wrapper
    melspec_fn_partial = functools.partial(python_dataset_melspectro_pipeline, model_name=model_name)
    # For TensorFlow functions (no wrapper needed)
    reshape_pipeline_partial = functools.partial(tensorflow_reshape_image_pipeline, model_name=model_name)
    shape_setter_partial = functools.partial(tensorflow_output_shape_setter, model_name=model_name)
    # Assuming tensorflow_image_augmentations might also need model_name in the future, prepare if needed:
    # image_augment_partial = functools.partial(tensorflow_image_augmentations, model_name=model_name)
    # -----------------------------------------------

    # Create the training dataset pipeline
    train_dataset = (train_ds
                     .shuffle(shuffle_buffer_size) # Use calculated or default buffer size
                     .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
                     .map(functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types), num_parallel_calls=parallel_calls)
                     .map(functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types), num_parallel_calls=parallel_calls)
                     .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
                     .map(functools.partial(python_fuction_wrapper, python_audio_augmentations, procs_output_types), num_parallel_calls=parallel_calls)
                     # Use the partial melspec function with the wrapper
                     .map(functools.partial(python_fuction_wrapper, melspec_fn_partial, procs_output_types), num_parallel_calls=parallel_calls)
                     # Use the partial reshape function directly (it's a TF function)
                     .map(reshape_pipeline_partial, num_parallel_calls=parallel_calls)
                     # Use image augmentations directly (assuming it's TF compatible and doesn't need model_name yet)
                     # If it needs model_name, use image_augment_partial
                     .map(tensorflow_image_augmentations, num_parallel_calls=parallel_calls)
                     .map(functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types), num_parallel_calls=parallel_calls)
                     .cache() # Cache after processing before shape setting/dropping extras
                     # Use the partial shape setter directly (it's a TF function)
                     .map(shape_setter_partial, num_parallel_calls=parallel_calls)
                     .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
                     .batch(SC['CLASSIFIER_BATCH_SIZE'])
                     .prefetch(parallel_calls)
                     )

    # Create the validation dataset pipeline (no audio/image augmentation)
    validation_dataset = (val_ds
                          .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
                          .map(functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types), num_parallel_calls=parallel_calls)
                          .map(functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types), num_parallel_calls=parallel_calls)
                          .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
                          # Use the partial melspec function with the wrapper
                          .map(functools.partial(python_fuction_wrapper, melspec_fn_partial, procs_output_types), num_parallel_calls=parallel_calls)
                          # Use the partial reshape function directly
                          .map(reshape_pipeline_partial, num_parallel_calls=parallel_calls)
                          # No image augmentation for validation
                          .map(functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types), num_parallel_calls=parallel_calls)
                          .cache()  # Use TensorFlow's in-memory cache
                          # Use the partial shape setter directly
                          .map(shape_setter_partial, num_parallel_calls=parallel_calls)
                          .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
                          .batch(SC['CLASSIFIER_BATCH_SIZE'])
                          .prefetch(parallel_calls)
                          )

    # Create the test dataset pipeline (no audio/image augmentation)
    test_dataset = (test_ds
                    .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types), num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
                    # Use the partial melspec function with the wrapper
                    .map(functools.partial(python_fuction_wrapper, melspec_fn_partial, procs_output_types), num_parallel_calls=parallel_calls)
                    # Use the partial reshape function directly
                    .map(reshape_pipeline_partial, num_parallel_calls=parallel_calls)
                    # No image augmentation for test
                    .map(functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types), num_parallel_calls=parallel_calls)
                    # Cache is optional for test set, depends on memory/repeated evaluation needs
                    # .cache()
                    # Use the partial shape setter directly
                    .map(shape_setter_partial, num_parallel_calls=parallel_calls)
                    .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
                    .batch(SC['CLASSIFIER_BATCH_SIZE'])
                    .prefetch(parallel_calls)
                    )

    # Use cardinality to print sizes safely
    print(f"Train dataset size (batches): {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation dataset size (batches): {tf.data.experimental.cardinality(validation_dataset).numpy()}")
    print(f"Test dataset size (batches): {tf.data.experimental.cardinality(test_dataset).numpy()}")
    print("Datasets built successfully!")

    return train_dataset, validation_dataset, test_dataset

# Example usage

if __name__ == "__main__":
    # Create datasets
    train_ds, val_ds, test_ds, class_names = create_datasets(audio_dir)
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Class names: {class_names}")


