"""This file contains all the functions related to 
loading and preprocessing the audio data

Look at optimised_engine_pipeline_MASTER.ipynb file for more information

"""


########################################################################################
# system constants
########################################################################################

SC = {
    'AUDIO_DATA_DIRECTORY': "C:\Project-Echo\src\Prototypes\data\Data Files\Bucket 3",
    'CACHE_DIRETORY': "C:\Project-Echo\src\Prototypes\engine\Optimising for Low Frequency Audio\cache",

    'AUDIO_CLIP_DURATION': 5, # seconds 
    'AUDIO_NFFT': 2048,
    'AUDIO_WINDOW': None,
    'AUDIO_STRIDE': 200,
    'AUDIO_SAMPLE_RATE': 48000,
    'AUDIO_MELS': 260,
    'AUDIO_FMIN': 20,
    'AUDIO_FMAX': 13000,
    'AUDIO_TOP_DB': 80,

    'MODEL_INPUT_IMAGE_WIDTH': 260,
    'MODEL_INPUT_IMAGE_HEIGHT': 260,
    'MODEL_INPUT_IMAGE_CHANNELS': 3,

    'USE_DISK_CACHE': True,
    'SAMPLE_VARIANTS': 20,
    'CLASSIFIER_BATCH_SIZE': 16,
    'MAX_EPOCHS': 30,
}


import tensorflow as tf
from pathlib import Path
import librosa as lb
import numpy as np
import functools
import diskcache as dc
import hashlib
import tensorflow_hub as hub
from pathlib import Path
import json

########################################################################################
# cache functions
########################################################################################

def enforce_memory_limit(mem_mb):
  # enforce memory limit on GPU

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(
          gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_mb)])
      print(f"vram limit set to {mem_mb}MB")
    except RuntimeError as e:
      print(e)


def tensorflow_add_variant_and_cache(path, label):
    variant     = tf.random.uniform(shape=(), minval=0, maxval=SC['SAMPLE_VARIANTS'], dtype=tf.int32)
    sample      = path
    cache_key   = b'no key'
    cache_found = np.int32(0)
    return sample, label, variant, cache_key, cache_found

def tensorflow_drop_variant_and_cache(sample, label, variant, cache_key, cache_found):
    return sample, label



def python_fuction_wrapper(pipeline_fn, out_types, sample, label, variant, cache_key, cache_found):

    # Use a lambda function to pass two arguments to the function
    sample, label, variant, cache_key, cache_found = tf.numpy_function(
        func=lambda v1,v2,v3,v4,v5: pipeline_fn(v1,v2,v3,v4,v5),
        inp=(sample, label, variant, cache_key, cache_found),
        Tout=out_types)

    return sample, label, variant, cache_key, cache_found

def python_disk_cache_start(sample, label, variant, cache_key, cache_found):

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



########################################################################################
# a helper function to create a hash key from a function signature and arguments
########################################################################################
def create_function_key(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    func_name = partial_func.func.__name__
    func_module = partial_func.func.__module__
    args_repr = repr(partial_func.args)
    kwargs_repr = repr(sorted(partial_func.keywords.items()))

    key = f"{func_module}.{func_name}:{args_repr}:{kwargs_repr}"
    # Use hashlib to create a hash of the key for shorter and consistent length
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    return key, key_hash, partial_func
      


########################################################################################
# create_datasets
########################################################################################

# Function to manually map paths to labels
def paths_and_labels_to_dataset(audio_paths, labels, num_classes):
    """
    Create a dataset from file paths and corresponding labels.
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
    """
    Walk through the directory and gather file paths and labels.
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

########################################################################################
# Create a DiskCache instance
# This cache will allow us store intermediate function results to speed up the 
# data processing pipeline
########################################################################################
if SC['USE_DISK_CACHE']:
    cache = dc.Cache(SC['CACHE_DIRETORY'], cull_limit=0, size_limit=10**9) 
# enforce max 5GB memory on GPU for this notebook if you have a small GPU
enforce_memory_limit(5120)

# Load class_names and labels from the JSON file
with open('class_names_labels.json', 'r') as f:
    data = json.load(f)

class_names = data['class_names']
labels = data['labels']
parallel_calls = tf.data.AUTOTUNE
cache_output_types = (tf.string,tf.float32,tf.int32,tf.string,tf.int32)
procs_output_types = (tf.float32,tf.float32,tf.int32,tf.string,tf.int32)


########################################################################################
# python_load_and_decode_file
########################################################################################

# enable this cache if using large audio files, and lots of available RAM
def python_load_and_decode_file(sample, label, variant, cache_key, cache_found):
    
    if cache_found == np.int32(0):
        
        tmp_audio_t = None
        
        with open(sample, 'rb') as file:

            # Load the audio data with librosa
            tmp_audio_t, _ = lb.load(file, sr=SC['AUDIO_SAMPLE_RATE'])
            
            # cast and keep right channel only
            if tmp_audio_t.ndim == 2 and tmp_audio_t.shape[0] == 2:
                tmp_audio_t = tmp_audio_t[1, :]
            
            # cast and keep right channel only
            tmp_audio_t = tmp_audio_t.astype(np.float32)
                    
            assert(tmp_audio_t is not None)
            assert(isinstance(tmp_audio_t, np.ndarray))
        
        sample = tmp_audio_t
        
    else:
        sample = cache[cache_key.decode('utf-8')]
    
    return sample, label, variant, cache_key, cache_found



def tensorflow_load_random_subsection(sample, label, variant, cache_key, cache_found):
    
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


########################################################################################
# mel_spectrogram_pipeline
########################################################################################

# this function takes the audio_clip and generates a melspectrogram representation
# the size of the image will always be the same as the length of audio is fixed and the
# various parameters such as AUDIO_STRIDE that impact image size are also fixed.
# this function takes the audio_clip and generates a melspectrogram representation
# the size of the image will always be the same as the length of audio is fixed and the
# various parameters such as AUDIO_STRIDE that impact image size are also fixed.
def python_dataset_melspectro_pipeline(sample, label, variant, cache_key, cache_found):
    
    if cache_found == np.int32(0):
        # Compute the mel-spectrogram
        image = lb.feature.melspectrogram(
            y=sample, 
            sr=SC['AUDIO_SAMPLE_RATE'], 
            n_fft=SC['AUDIO_NFFT'], 
            hop_length=SC['AUDIO_STRIDE'], 
            n_mels=SC['AUDIO_MELS'],
            fmin=SC['AUDIO_FMIN'],
            fmax=SC['AUDIO_FMAX'],
            win_length=SC['AUDIO_WINDOW'])

        # Optionally convert the mel-spectrogram to decibel scale
        image = lb.power_to_db(
            image, 
            top_db=SC['AUDIO_TOP_DB'], 
            ref=1.0)
        
        # Calculate the expected number of samples in a clip
        expected_clip_samples = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / SC['AUDIO_STRIDE'])
        
        # swap axis and clip to expected samples to avoid rounding errors
        image = np.moveaxis(image, 1, 0)
        sample = image[0:expected_clip_samples,:]
    
    return sample, label, variant, cache_key, cache_found


def tensorflow_reshape_image_pipeline(sample, label, variant, cache_key, cache_found):   
    
    if cache_found == np.int32(0):
        # reshape into standard 3 channels to add the color channel
        image = tf.expand_dims(sample, -1)
        
        # most pre-trained model classifer model expects 3 color channels
        image = tf.repeat(image, SC['MODEL_INPUT_IMAGE_CHANNELS'], axis=2)
        
        # Calculate the expected number of samples in a clip
        expected_clip_samples = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / SC['AUDIO_STRIDE'])
        
        # calculate the image shape and ensure it is correct   
        image = tf.ensure_shape(image, [expected_clip_samples, SC['AUDIO_MELS'], SC['MODEL_INPUT_IMAGE_CHANNELS']])
        
        # note here a high quality LANCZOS5 is applied to resize the image to match model image input size
        image = tf.image.resize(image, (SC['MODEL_INPUT_IMAGE_WIDTH'],SC['MODEL_INPUT_IMAGE_HEIGHT']), 
                                method=tf.image.ResizeMethod.LANCZOS5)

        # rescale to range [0,1]
        image = image - tf.reduce_min(image) 
        sample = image / (tf.reduce_max(image)+0.0000001)
    
    return sample, label, variant, cache_key, cache_found

def tensorflow_output_shape_setter(sample, label, variant, cache_key, cache_found):
    global class_names
    sample.set_shape([SC['MODEL_INPUT_IMAGE_WIDTH'], SC['MODEL_INPUT_IMAGE_HEIGHT'], SC['MODEL_INPUT_IMAGE_CHANNELS']])
    label.set_shape([len(class_names),]) 
    return sample, label, variant, cache_key, cache_found

########################################################################################
# create_datasets_pipeline
########################################################################################

# data prep
def data_init(data_dir):
    train_ds, val_ds, test_ds, class_names = create_datasets(data_dir,train_split=0.8, val_split=0.19)
    len_train_ds = len(train_ds)
    parallel_calls = tf.data.AUTOTUNE
    cache_output_types = (tf.string,tf.float32,tf.int32,tf.string,tf.int32)
    procs_output_types = (tf.float32,tf.float32,tf.int32,tf.string,tf.int32)
    return train_ds, val_ds, test_ds, class_names, len_train_ds, parallel_calls, cache_output_types, procs_output_types



# load train dataset
def load_train_dataset(train_ds, len_train_ds, parallel_calls, cache_output_types, procs_output_types):
    train_dataset = (train_ds
                    .shuffle(len_train_ds)
                    .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types), num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
                    # .map(functools.partial(python_fuction_wrapper, python_audio_augmentations, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_dataset_melspectro_pipeline, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
                    # .map(tensorflow_image_augmentations, num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
                    .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
                    .batch(SC['CLASSIFIER_BATCH_SIZE'])
                    .prefetch(parallel_calls)
                    .repeat(count=1)
    )
    return train_dataset

# load val dataset
def load_val_dataset(val_ds, parallel_calls, cache_output_types, procs_output_types):
    val_dataset = (val_ds
                        .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
                        .map(functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types), num_parallel_calls=parallel_calls)
                        .map(functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types), num_parallel_calls=parallel_calls)
                        .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
                        .map(functools.partial(python_fuction_wrapper, python_dataset_melspectro_pipeline, procs_output_types), num_parallel_calls=parallel_calls)
                        .map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
                        .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
                        .map(functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types), num_parallel_calls=parallel_calls)
                        .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
                        .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
                        .batch(SC['CLASSIFIER_BATCH_SIZE'])
                        .prefetch(parallel_calls)
                        .repeat(count=1)
    )
    return val_dataset

def load_test_dataset(test_ds, parallel_calls, cache_output_types, procs_output_types):
    test_dataset = (test_ds
                    .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types), num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_dataset_melspectro_pipeline, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
                    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
                    .map(functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types), num_parallel_calls=parallel_calls)
                    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
                    .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
                    .batch(SC['CLASSIFIER_BATCH_SIZE'])
                    .prefetch(parallel_calls)
                    .repeat(count=1)
    )
    return test_dataset

########################################################################################
# model creation
########################################################################################

# Custom layer to wrap hub.KerasLayer
class HubLayer(tf.keras.layers.Layer):
    def __init__(self, hub_url, trainable=True):
        super(HubLayer, self).__init__()
        self.hub_layer = hub.KerasLayer(hub_url, trainable=trainable)

    def call(self, inputs):
        return self.hub_layer(inputs)

def build_model(trainable):
    # Build a classification model using a pre-trained EfficientNetV2
    model = tf.keras.Sequential(
        [
            # Input layer with specified image dimensions
            tf.keras.layers.InputLayer(input_shape=(SC['MODEL_INPUT_IMAGE_HEIGHT'], 
                                                    SC['MODEL_INPUT_IMAGE_WIDTH'], 
                                                    SC['MODEL_INPUT_IMAGE_CHANNELS'])),

            # Use the EfficientNetV2 model as a feature generator (needs 260x260x3 images)
            HubLayer("https://www.kaggle.com/models/google/efficientnet-v2/TensorFlow2/imagenet1k-b0-classification/2", trainable),

            # Add the classification layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),

            # Fully connected layer with multiple of the number of classes
            tf.keras.layers.Dense(len(class_names) * 8,
                                  activation="relu"),
            tf.keras.layers.BatchNormalization(),

            # Another fully connected layer with multiple of the number of classes
            tf.keras.layers.Dense(len(class_names) * 4,
                                  activation="relu"),
            tf.keras.layers.BatchNormalization(),

            # Add dropout to reduce overfitting
            tf.keras.layers.Dropout(0.50),

            # Output layer with one node per class, without activation
            tf.keras.layers.Dense(len(class_names), activation=None),
        ]
    )
    # Set the input shape for the model
    model.build([None, 
                 SC['MODEL_INPUT_IMAGE_HEIGHT'],
                 SC['MODEL_INPUT_IMAGE_WIDTH'], 
                 SC['MODEL_INPUT_IMAGE_CHANNELS']])

    # Display the model summary
    model.summary()

    return model