
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
import librosa as lb
import audiomentations as augs


SC = {
    'AUDIO_DATA_DIRECTORY': "C:/Users/regin/Documents/SIT374/train_data/b3", #original dataset location.
    'MODEL_DIRECTORY': "C:/Users/regin/Documents/SIT374/Project-Echo/src/Components/Engine/models/echo_model/1/", #directory of the saved_model.pb file
    'CLASS_NAMES_DIRECTORY': "class_names_labels.json", #class names and labels JSON file
    
    # "C:/Users/regin/Documents/SIT374/Project-Echo/src/Components/Engine/models/echo_model/1/"


    'AUDIO_CLIP_DURATION': 5, # seconds 
    'AUDIO_NFFT': 2048,
    'AUDIO_WINDOW': None,
    'AUDIO_STRIDE': 200,
    'AUDIO_SAMPLE_RATE': 44000,
    'AUDIO_MELS': 260,
    'AUDIO_FMIN': 20,
    'AUDIO_FMAX': 13000,
    'AUDIO_TOP_DB': 80,


    'MODEL_INPUT_IMAGE_WIDTH': 260,
    'MODEL_INPUT_IMAGE_HEIGHT': 260,
    'MODEL_INPUT_IMAGE_CHANNELS': 3,

    'MAX_EPOCHS': 10
}

def update_global_config(new_config):
    """
    Update the global configuration `SC` with values from `new_config`.
    """
    global SC
    SC.update(new_config)
    return SC

def default_config():
    """
    Return the default configuration.
    """
    default_SC = {
        'AUDIO_DATA_DIRECTORY': "C:/Users/regin/Documents/SIT374/train_data/b3", #original dataset location.
        'MODEL_DIRECTORY': "C:/Users/regin/Documents/SIT374/Project-Echo/src/Components/Engine/models/echo_model/1/", #directory of the saved_model.pb file
        'CLASS_NAMES_DIRECTORY': "class_names_labels.json", #class names and labels JSON file
    
        # "C:/Users/regin/Documents/SIT374/Project-Echo/src/Components/Engine/models/echo_model/1/"


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

        'MAX_EPOCHS': 10
    }
    SC.update(default_SC)


### and class names ###
def species_list(directory):
    """
    Walk through the directory and gather file paths and labels.
    """
    labels = []
    class_names = sorted([dir.name for dir in Path(directory).glob('*') if dir.is_dir()])
    

    for label, class_name in enumerate(class_names):
        class_dir = Path(directory) / class_name
        for file_path in class_dir.glob(f'**/*'):
            labels.append(label)  # Store the integer label instead of the class name

    return labels, class_names

labels, class_names = species_list(SC['AUDIO_DATA_DIRECTORY'])

### Getting the paths, labels, and creating dataset ###
def audio_files(directory, samples, file_types=('.ogg', '.mp3', '.wav', '.flac')):
    """
    Walk through the directory, group by class, select a limited number, and gather file paths and labels.
    """
    audio_files = []
    labels = []
    class_names = sorted([dir.name for dir in Path(directory).glob('*') if dir.is_dir()])
    

    for label, class_name in enumerate(class_names):
        class_dir = Path(directory) / class_name

        class_files = [
            str(file_path) 
            for file_path in class_dir.glob(f'**/*') 
            if file_path.suffix.lower() in file_types
        ]

        # Randomly select up to `samples_per_class` files from this class
        selected_files = random.sample(class_files, min(samples, len(class_files)))
        
        # Append selected files and their corresponding labels
        audio_files.extend(selected_files)
        labels.extend([label] * len(selected_files))


    return audio_files, labels

def paths_and_labels_to_dataset(audio_paths, labels, num_classes):
    """
    Create a tensorflow dataset from file paths and corresponding labels.
    """
    # Convert labels to one-hot encoded format using tf.one_hot
    one_hot_labels = tf.one_hot(labels, depth=num_classes)

    # Create TensorFlow datasets for the paths and labels
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(one_hot_labels)

    # Zip both datasets into a single dataset
    zipped_path_ds = tf.data.Dataset.zip((path_ds, label_ds))
    
    return zipped_path_ds

def create_datasets(audio_dir, class_names, samples):
    """
    Create dataset from audio directory.
    """
    # Index the directory and get file paths and labels
    file_paths, labels = audio_files(audio_dir, samples=samples)

    # Convert paths and labels to a TensorFlow dataset
    dataset = paths_and_labels_to_dataset(
        audio_paths=file_paths,
        labels=labels,
        num_classes=len(class_names)
    )
    
    # Calculate the size of the dataset
    dataset_size = len(file_paths)
    

    # Split the dataset
    dataset = dataset.take(dataset_size)

    return dataset



def split_dataset(dataset, train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Splits a dataset into training, validation, and testing datasets.

    Args:
        dataset: tf.data.Dataset object to split.
        train_split (float): Fraction of the dataset for training.
        val_split (float): Fraction of the dataset for validation.
        test_split (float): Fraction of the dataset for testing.

    Returns:
        train_dataset, val_dataset, test_dataset: tf.data.Dataset splits.
    """
    # assert train_split + val_split + test_split == 1.0, "Splits must sum to 1."

    # Calculate dataset sizes
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    # Shuffle the dataset and split
    dataset = dataset.shuffle(dataset_size, seed=223740481)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset








###################################################
# Convert data to mel spectrograph
####################################################

def python_function_wrapper(pipeline_fn, out_types, sample, label):

    # Use a lambda function to pass two arguments to the function
    sample, label= tf.numpy_function(
        func=lambda v1,v2: pipeline_fn(v1,v2),
        inp=(sample, label),
        Tout=out_types)

    return sample, label


def python_load_and_decode_file(sample, label):
            
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
    
    return sample, label



def tensorflow_load_random_subsection(sample, label):
    '''
    Extracts a random subsection of an audio sample with a fixed duration (AUDIO_CLIP_DURATION).
    If the sample is shorter than the required duration, it pads the audio with silence to make it the correct length.
    '''
    global SC
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

    return sample, label


#python_audio_augmentations
# Audio augmentation pipeline
def python_audio_augmentations(sample, label):
    '''
    Applies data augmentation techniques to the audio to improve the model's generalization.
    Uses the audiomentations library to perform:
    Adding Gaussian noise.
    Time-stretching (changing playback speed).
    Pitch-shifting (altering pitch).
    Shifting audio in time.
    '''

    # See https://github.com/iver56/audiomentations for more options
    augmentations = augs.Compose([
        # Add Gaussian noise with a random amplitude to the audio
        # This can help the model generalize to real-world scenarios where noise is present
        augs.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),

        # Time-stretch the audio without changing its pitch
        # This can help the model become invariant to small changes in the speed of the audio
        augs.TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),

        # Shift the pitch of the audio within a range of semitones
        # This can help the model generalize to variations in pitch that may occur in real-world scenarios
        augs.PitchShift(min_semitones=-4, max_semitones=4, p=0.2),

        # Shift the audio in time by a random fraction
        # This can help the model become invariant to the position of important features in the audio
        augs.Shift(min_shift=-0.5, max_shift=0.5, p=0.2),
    ])
    
    # apply audio augmentation to the clip
    # note: this augmentation is NOT applied in the test and validation pipelines
    sample = augmentations(samples=sample, sample_rate=SC['AUDIO_SAMPLE_RATE'])

    return sample, label


# this function takes the audio_clip and generates a melspectrogram representation
# the size of the image will always be the same as the length of audio is fixed and the
# various parameters such as AUDIO_STRIDE that impact image size are also fixed.
# this function takes the audio_clip and generates a melspectrogram representation
# the size of the image will always be the same as the length of audio is fixed and the
# various parameters such as AUDIO_STRIDE that impact image size are also fixed.
def python_dataset_melspectro_pipeline(sample, label):
    '''
    Converts an audio sample into its mel spectrogram representation, which is a 2D image-like representation of sound.
    The mel spectrogram is resized to a fixed size to ensure consistency.
    '''

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
    
    return sample, label

def tensorflow_reshape_image_pipeline(sample, label):   
    '''
    Expand the spectrogram to have a single grayscale channel.
    Repeat the channel to simulate 3-color channels (as required by most pre-trained models).
    Resize the image to the model's input size using high-quality resizing (LANCZOS5).
    Normalize the pixel values to the range [0,1].
    '''

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

    return sample, label



def tensorflow_output_shape_setter(sample, label):
    '''
    Ensures the output sample and label shapes are explicitly set for TensorFlow processing.
    Set the shape of the spectrogram image (sample) to match the model's input size.
    Set the label shape to match the number of classes.
    '''
    global class_names
    sample.set_shape([SC['MODEL_INPUT_IMAGE_WIDTH'], SC['MODEL_INPUT_IMAGE_HEIGHT'], SC['MODEL_INPUT_IMAGE_CHANNELS']])
    label.set_shape([len(class_names),]) 
    return sample, label



