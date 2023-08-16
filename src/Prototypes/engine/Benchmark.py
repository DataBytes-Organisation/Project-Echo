# This python file is used to compare the performance of 2 trained models.
# After the prediction, it will print the value of precision, loss, recall, fscore and a confusion matrix for you to compare the performance between the 2 models


import warnings
from platform import python_version

warnings.filterwarnings("ignore")
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import tensorflow as tf
import tensorflow_addons as tfa
from keras.utils import dataset_utils
import librosa
import audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import functools
import hashlib
import diskcache as dc

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

print('Python Version           : ', python_version())
print('TensorFlow Version       : ', tf.__version__)
print('Librosa Version          : ', librosa.__version__)
print('Audiomentations Version  : ', audiomentations.__version__)
SC = {
    'AUDIO_DATA_DIRECTORY': "E:\Training_Data\ProjectEcho\project_echo_bucket_2",
    'CACHE_DIRETORY': "E:\Training_Cache",

    'AUDIO_CLIP_DURATION': 5,  # seconds
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
    'MAX_EPOCHS': 1,
}


def enforce_memory_limit(mem_mb):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_mb)])
            print(f"vram limit set to {mem_mb}MB")
        except RuntimeError as e:
            print(e)


if SC['USE_DISK_CACHE']:
    cache = dc.Cache(SC['CACHE_DIRETORY'], cull_limit=0, size_limit=10 ** 9)


def create_function_key(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    func_name = partial_func.func.__name__
    func_module = partial_func.func.__module__
    args_repr = repr(partial_func.args)
    kwargs_repr = repr(sorted(partial_func.keywords.items()))
    key = f"{func_module}.{func_name}:{args_repr}:{kwargs_repr}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash, partial_func


def paths_and_labels_to_dataset(image_paths, labels, num_classes):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = dataset_utils.labels_to_dataset(
        labels,
        'categorical',
        num_classes)
    zipped_path_ds = tf.data.Dataset.zip((path_ds, label_ds))
    return zipped_path_ds


def create_datasets(audio_files, train_split=0.2, val_split=0.3):
    file_paths, labels, class_names = dataset_utils.index_directory(
        audio_files,
        labels="inferred",
        formats=('.ogg', '.mp3', '.wav', '.flac'),
        class_names=None,
        shuffle=True,
        seed=42,
        follow_links=False)
    dataset = paths_and_labels_to_dataset(
        image_paths=file_paths,
        labels=labels,
        num_classes=len(class_names))
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size).take(test_size)
    return train_ds, val_ds, test_ds, class_names


train_ds, val_ds, test_ds, class_names = create_datasets(SC['AUDIO_DATA_DIRECTORY'], train_split=0.8, val_split=0.19)
print("Class names: ", class_names)
print(f"Training   dataset length: {len(train_ds)}")
print(f"Validation dataset length: {len(val_ds)}")
print(f"Test       dataset length: {len(test_ds)}")
for item in train_ds.take(10):
    print(item)


def python_load_and_decode_file(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        tmp_audio_t = None
        with open(sample, 'rb') as file:
            tmp_audio_t, _ = librosa.load(file, sr=SC['AUDIO_SAMPLE_RATE'])
            if tmp_audio_t.ndim == 2 and tmp_audio_t.shape[0] == 2:
                tmp_audio_t = tmp_audio_t[1, :]
            tmp_audio_t = tmp_audio_t.astype(np.float32)
            assert (tmp_audio_t is not None)
            assert (isinstance(tmp_audio_t, np.ndarray))
        sample = tmp_audio_t
    else:
        sample = cache[cache_key.decode('utf-8')]
    return sample, label, variant, cache_key, cache_found


def tensorflow_load_random_subsection(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        duration_secs = SC['AUDIO_CLIP_DURATION']
        audio_duration_secs = tf.shape(sample)[0] / SC['AUDIO_SAMPLE_RATE']
        if audio_duration_secs > duration_secs:
            max_start = tf.cast(audio_duration_secs - duration_secs, tf.float32)
            start_time_secs = tf.random.uniform((), 0.0, max_start, dtype=tf.float32)
            start_index = tf.cast(start_time_secs * SC['AUDIO_SAMPLE_RATE'], dtype=tf.int32)
            end_index = tf.cast(start_index + tf.cast(duration_secs, tf.int32) * SC['AUDIO_SAMPLE_RATE'], tf.int32)
            subsection = sample[start_index: end_index]

        else:
            padding_length = duration_secs * SC['AUDIO_SAMPLE_RATE'] - tf.shape(sample)[0]
            padding = tf.zeros([padding_length], dtype=sample.dtype)
            subsection = tf.concat([sample, padding], axis=0)

        sample = subsection

    return sample, label, variant, cache_key, cache_found


def python_audio_augmentations(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        augmentations = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
        ])
        sample = augmentations(samples=sample, sample_rate=SC['AUDIO_SAMPLE_RATE'])
    return sample, label, variant, cache_key, cache_found


def tensorflow_image_augmentations(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        degrees = tf.random.uniform(shape=(1,), minval=-2, maxval=2)
        radians = degrees * 0.017453292519943295
        sample = tfa.image.rotate(sample, radians, interpolation='bilinear')
    return sample, label, variant, cache_key, cache_found


def python_dataset_melspectro_pipeline(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        image = librosa.feature.melspectrogram(
            y=sample,
            sr=SC['AUDIO_SAMPLE_RATE'],
            n_fft=SC['AUDIO_NFFT'],
            hop_length=SC['AUDIO_STRIDE'],
            n_mels=SC['AUDIO_MELS'],
            fmin=SC['AUDIO_FMIN'],
            fmax=SC['AUDIO_FMAX'],
            win_length=SC['AUDIO_WINDOW'])
        image = librosa.power_to_db(
            image,
            top_db=SC['AUDIO_TOP_DB'],
            ref=1.0)
        expected_clip_samples = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / SC['AUDIO_STRIDE'])
        image = np.moveaxis(image, 1, 0)
        sample = image[0:expected_clip_samples, :]
    return sample, label, variant, cache_key, cache_found


def tensorflow_reshape_image_pipeline(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        image = tf.expand_dims(sample, -1)
        image = tf.repeat(image, SC['MODEL_INPUT_IMAGE_CHANNELS'], axis=2)
        expected_clip_samples = int(SC['AUDIO_CLIP_DURATION'] * SC['AUDIO_SAMPLE_RATE'] / SC['AUDIO_STRIDE'])
        image = tf.ensure_shape(image, [expected_clip_samples, SC['AUDIO_MELS'], SC['MODEL_INPUT_IMAGE_CHANNELS']])
        image = tf.image.resize(image, (SC['MODEL_INPUT_IMAGE_WIDTH'], SC['MODEL_INPUT_IMAGE_HEIGHT']),
                                method=tf.image.ResizeMethod.LANCZOS5)
        image = image - tf.reduce_min(image)
        sample = image / (tf.reduce_max(image) + 0.0000001)
    return sample, label, variant, cache_key, cache_found


def tensorflow_add_variant_and_cache(path, label):
    variant = tf.random.uniform(shape=(), minval=0, maxval=SC['SAMPLE_VARIANTS'], dtype=tf.int32)
    sample = path
    cache_key = b'no key'
    cache_found = np.int32(0)
    return sample, label, variant, cache_key, cache_found


def tensorflow_drop_variant_and_cache(sample, label, variant, cache_key, cache_found):
    return sample, label


def tensorflow_output_shape_setter(sample, label, variant, cache_key, cache_found):
    sample.set_shape([SC['MODEL_INPUT_IMAGE_WIDTH'], SC['MODEL_INPUT_IMAGE_HEIGHT'], SC['MODEL_INPUT_IMAGE_CHANNELS']])
    label.set_shape([len(class_names), ])
    return sample, label, variant, cache_key, cache_found


def python_fuction_wrapper(pipeline_fn, out_types, sample, label, variant, cache_key, cache_found):
    sample, label, variant, cache_key, cache_found = tf.numpy_function(
        func=lambda v1, v2, v3, v4, v5: pipeline_fn(v1, v2, v3, v4, v5),
        inp=(sample, label, variant, cache_key, cache_found),
        Tout=out_types)
    return sample, label, variant, cache_key, cache_found


def python_disk_cache_start(sample, label, variant, cache_key, cache_found):
    cache_key = b'no key'
    cache_found = np.int32(0)
    if SC['USE_DISK_CACHE']:
        _, cache_key, _ = create_function_key(python_disk_cache_start, sample, label, variant)
        if cache_key in cache:
            cache_found = np.int32(1)
        else:
            pass
    return sample, label, variant, cache_key, cache_found


def python_disk_cache_end(sample, label, variant, cache_key, cache_found):
    cache_key = cache_key.decode('utf-8')
    if SC['USE_DISK_CACHE']:
        if cache_found == np.int32(0):
            cache[cache_key] = sample

    return sample, label, variant, cache_key, cache_found


# Set the parallel calls
AUTOTUNE = tf.data.AUTOTUNE
parallel_calls = AUTOTUNE

# Create the datasets
train_dataset, validation_dataset, test_dataset, class_names = create_datasets(SC['AUDIO_DATA_DIRECTORY'])

# Preprocess the test dataset
cache_output_types = (tf.string, tf.float32, tf.int32, tf.string, tf.int32)
procs_output_types = (tf.float32, tf.float32, tf.int32, tf.string, tf.int32)

test_dataset = (test_dataset
                .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
                .map(functools.partial(python_fuction_wrapper, python_disk_cache_start, cache_output_types),
                     num_parallel_calls=parallel_calls)
                .map(functools.partial(python_fuction_wrapper, python_load_and_decode_file, procs_output_types),
                     num_parallel_calls=parallel_calls)
                .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
                .map(functools.partial(python_fuction_wrapper, python_dataset_melspectro_pipeline, procs_output_types),
                     num_parallel_calls=parallel_calls)
                .map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
                .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
                .map(functools.partial(python_fuction_wrapper, python_disk_cache_end, procs_output_types),
                     num_parallel_calls=parallel_calls)
                .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
                .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
                .batch(SC['CLASSIFIER_BATCH_SIZE'])
                .prefetch(parallel_calls)
                .repeat(count=1)
                )

# Load your first model here
model1 = tf.keras.models.load_model('models/echo_model/1/')

# Evaluate the model
y1_true = []
y1_pred = []
for batch_images, batch_labels in test_dataset:
    predictions = model1.predict(batch_images)
    y1_true.extend(np.argmax(batch_labels, axis=-1))
    y1_pred.extend(np.argmax(predictions, axis=-1))

# Calculate the metrics
precision1, recall1, fscore1, _ = precision_recall_fscore_support(y1_true, y1_pred, average='macro')
confusion1 = confusion_matrix(y1_true, y1_pred)
loss1 = model1.evaluate(test_dataset, return_dict=True)["loss"]

print(f'Precision: {precision1}')
print(f'Loss: {loss1}')
print(f'Recall: {recall1}')
print(f'F-score: {fscore1}')
print(f'Confusion Matrix:\n{confusion1}')

# Load your second model here
model2 = tf.keras.models.load_model('models/Xception/')

y2_true = []
y2_pred = []
for batch_images, batch_labels in test_dataset:
    predictions = model2.predict(batch_images)
    y2_true.extend(np.argmax(batch_labels, axis=-1))
    y2_pred.extend(np.argmax(predictions, axis=-1))

# Calculate the metrics
precision2, recall2, fscore2, _ = precision_recall_fscore_support(y2_true, y2_pred, average='macro')
confusion2 = confusion_matrix(y2_true, y2_pred)
loss2 = model2.evaluate(test_dataset, return_dict=True)["loss"]

print(f'Precision: {precision2}')
print(f'Loss: {loss2}')
print(f'Recall: {recall2}')
print(f'F-score: {fscore2}')
print(f'Confusion Matrix:\n{confusion2}')
