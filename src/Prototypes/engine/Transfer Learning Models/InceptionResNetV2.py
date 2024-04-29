import functools
import hashlib
import os
import time
import warnings
from platform import python_version

import audiomentations
import chardet
import diskcache as dc
import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
from audiomentations import AddGaussianNoise, Compose, PitchShift, Shift, TimeStretch
from keras.applications import InceptionResNetV2
from keras.utils import dataset_utils
from tensorflow import keras

# print system information
print("Python Version           : ", python_version())
print("TensorFlow Version       : ", tf.__version__)
print("Keras Version            : ", keras.__version__)
print("Librosa Version          : ", librosa.__version__)
print("Audiomentations Version  : ", audiomentations.__version__)

warnings.filterwarnings("ignore")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from _model_config import *


def enforce_memory_limit(mem_mb):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=mem_mb
                    )
                ],
            )
            print(f"vram limit set to {mem_mb}MB")
        except RuntimeError as e:
            print(e)


# enforce max 5GB memory on GPU for this notebook if you have a small GPU
# enforce_memory_limit(5120)

if SC["USE_DISK_CACHE"]:
    cache = dc.Cache(SC["CACHE_DIRETORY"], cull_limit=0, size_limit=10**9)


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
    label_ds = dataset_utils.labels_to_dataset(labels, "categorical", num_classes)
    zipped_path_ds = tf.data.Dataset.zip((path_ds, label_ds))
    return zipped_path_ds


def create_datasets(audio_files, train_split=0.7, val_split=0.2):
    file_paths, labels, class_names = dataset_utils.index_directory(
        audio_files,
        labels="inferred",
        formats=(".ogg", ".mp3", ".wav", ".flac"),
        class_names=None,
        shuffle=True,
        seed=42,
        follow_links=False,
    )
    dataset = paths_and_labels_to_dataset(
        image_paths=file_paths, labels=labels, num_classes=len(class_names)
    )
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size).take(test_size)
    return train_ds, val_ds, test_ds, class_names


train_ds, val_ds, test_ds, class_names = create_datasets(
    SC["AUDIO_DATA_DIRECTORY"], train_split=0.8, val_split=0.19
)
print("Class names: ", class_names)
print(f"Training dataset length: {len(train_ds)}")
print(f"Validation dataset length: {len(val_ds)}")
print(f"Test dataset length: {len(test_ds)}")


def python_load_and_decode_file(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        tmp_audio_t = None
        with open(sample, "rb") as file:
            tmp_audio_t, _ = librosa.load(file, sr=SC["AUDIO_SAMPLE_RATE"])
            if tmp_audio_t.ndim == 2 and tmp_audio_t.shape[0] == 2:
                tmp_audio_t = tmp_audio_t[1, :]
            tmp_audio_t = tmp_audio_t.astype(np.float32)

            assert tmp_audio_t is not None
            assert isinstance(tmp_audio_t, np.ndarray)
        sample = tmp_audio_t
    else:
        sample = cache[cache_key.decode("utf-8")]
    return sample, label, variant, cache_key, cache_found


def tensorflow_load_random_subsection(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        duration_secs = SC["AUDIO_CLIP_DURATION"]
        audio_duration_secs = tf.shape(sample)[0] / SC["AUDIO_SAMPLE_RATE"]
        if audio_duration_secs > duration_secs:
            max_start = tf.cast(audio_duration_secs - duration_secs, tf.float32)
            start_time_secs = tf.random.uniform((), 0.0, max_start, dtype=tf.float32)
            start_index = tf.cast(
                start_time_secs * SC["AUDIO_SAMPLE_RATE"], dtype=tf.int32
            )
            end_index = tf.cast(
                start_index
                + tf.cast(duration_secs, tf.int32) * SC["AUDIO_SAMPLE_RATE"],
                tf.int32,
            )
            subsection = sample[start_index:end_index]
        else:
            padding_length = (
                duration_secs * SC["AUDIO_SAMPLE_RATE"] - tf.shape(sample)[0]
            )
            padding = tf.zeros([padding_length], dtype=sample.dtype)
            subsection = tf.concat([sample, padding], axis=0)
        sample = subsection
    return sample, label, variant, cache_key, cache_found


def python_audio_augmentations(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        # See https://github.com/iver56/audiomentations for more options
        augmentations = Compose(
            [
                # Add Gaussian noise with a random amplitude to the audio
                # This can help the model generalize to real-world scenarios where noise is present
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
                # Time-stretch the audio without changing its pitch
                # This can help the model become invariant to small changes in the speed of the audio
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
                # Shift the pitch of the audio within a range of semitones
                # This can help the model generalize to variations in pitch that may occur in real-world scenarios
                PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                # # Shift the audio in time by a random fraction
                # # This can help the model become invariant to the position of important features in the audio
                # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
            ]
        )
        sample = augmentations(samples=sample, sample_rate=SC["AUDIO_SAMPLE_RATE"])
    return sample, label, variant, cache_key, cache_found


def tensorflow_image_augmentations(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        # random rotation -2 deg to 2 deg
        degrees = tf.random.uniform(shape=(1,), minval=-2, maxval=2)
        # convert the angle in degree to radians
        radians = degrees * 0.017453292519943295
        # rotate the image
        sample = tfa.image.rotate(sample, radians, interpolation="bilinear")
    return sample, label, variant, cache_key, cache_found


def python_dataset_melspectro_pipeline(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        image = librosa.feature.melspectrogram(
            y=sample,
            sr=SC["AUDIO_SAMPLE_RATE"],
            n_fft=SC["AUDIO_NFFT"],
            hop_length=SC["AUDIO_STRIDE"],
            n_mels=SC["AUDIO_MELS"],
            fmin=SC["AUDIO_FMIN"],
            fmax=SC["AUDIO_FMAX"],
            win_length=SC["AUDIO_WINDOW"],
        )
        image = librosa.power_to_db(image, top_db=SC["AUDIO_TOP_DB"], ref=1.0)
        expected_clip_samples = int(
            SC["AUDIO_CLIP_DURATION"] * SC["AUDIO_SAMPLE_RATE"] / SC["AUDIO_STRIDE"]
        )
        image = np.moveaxis(image, 1, 0)
        sample = image[0:expected_clip_samples, :]
    return sample, label, variant, cache_key, cache_found


def tensorflow_reshape_image_pipeline(sample, label, variant, cache_key, cache_found):
    if cache_found == np.int32(0):
        # reshape into standard 3 channels to add the color channel
        image = tf.expand_dims(sample, -1)
        # most pre-trained model classifer model expects 3 color channels
        image = tf.repeat(image, SC["MODEL_INPUT_IMAGE_CHANNELS"], axis=2)
        # Calculate the expected number of samples in a clip
        expected_clip_samples = int(
            SC["AUDIO_CLIP_DURATION"] * SC["AUDIO_SAMPLE_RATE"] / SC["AUDIO_STRIDE"]
        )
        # calculate the image shape and ensure it is correct
        image = tf.ensure_shape(
            image,
            [expected_clip_samples, SC["AUDIO_MELS"], SC["MODEL_INPUT_IMAGE_CHANNELS"]],
        )
        # note here a high quality LANCZOS5 is applied to resize the image to match model image input size
        image = tf.image.resize(
            image,
            (SC["MODEL_INPUT_IMAGE_WIDTH"], SC["MODEL_INPUT_IMAGE_HEIGHT"]),
            method=tf.image.ResizeMethod.LANCZOS5,
        )
        # rescale to range [0,1]
        image = image - tf.reduce_min(image)
        sample = image / (tf.reduce_max(image) + 0.0000001)
    return sample, label, variant, cache_key, cache_found


def tensorflow_add_variant_and_cache(path, label):
    variant = tf.random.uniform(
        shape=(), minval=0, maxval=SC["SAMPLE_VARIANTS"], dtype=tf.int32
    )
    sample = path
    cache_key = b"no key"
    cache_found = np.int32(0)
    return sample, label, variant, cache_key, cache_found


def tensorflow_drop_variant_and_cache(sample, label, variant, cache_key, cache_found):
    return sample, label


def tensorflow_output_shape_setter(sample, label, variant, cache_key, cache_found):
    sample.set_shape(
        [
            SC["MODEL_INPUT_IMAGE_WIDTH"],
            SC["MODEL_INPUT_IMAGE_HEIGHT"],
            SC["MODEL_INPUT_IMAGE_CHANNELS"],
        ]
    )
    label.set_shape(
        [
            len(class_names),
        ]
    )
    return sample, label, variant, cache_key, cache_found


def python_fuction_wrapper(
    pipeline_fn, out_types, sample, label, variant, cache_key, cache_found
):
    # Use a lambda function to pass two arguments to the function
    sample, label, variant, cache_key, cache_found = tf.numpy_function(
        func=lambda v1, v2, v3, v4, v5: pipeline_fn(v1, v2, v3, v4, v5),
        inp=(sample, label, variant, cache_key, cache_found),
        Tout=out_types,
    )
    return sample, label, variant, cache_key, cache_found


def python_disk_cache_start(sample, label, variant, cache_key, cache_found):
    cache_key = b"no key"
    cache_found = np.int32(0)
    if SC["USE_DISK_CACHE"]:
        _, cache_key, _ = create_function_key(
            python_disk_cache_start, sample, label, variant
        )
        if cache_key in cache:
            # print(f'found {cache_key} in cache')
            cache_found = np.int32(1)
        else:
            pass
            # print(f'{cache_key} not found in cache')
    return sample, label, variant, cache_key, cache_found


def python_disk_cache_end(sample, label, variant, cache_key, cache_found):
    cache_key = cache_key.decode("utf-8")
    if SC["USE_DISK_CACHE"]:
        # if it was not found in the cache at the start, then populate with what we built
        # during the pipeline execution
        if cache_found == np.int32(0):
            # print(f'adding {cache_key} to cache')
            cache[cache_key] = sample
        # else:
        #    sample = cache[cache_key]
    return sample, label, variant, cache_key, cache_found


########################################################################################
# Create the datasets necessary for training a classification model
# Note: python and tensorflow functions are treated differently in the tensorflow
# pipeline.  Each python function needs to be wrapped.
# this is why each pipeline function starts with python_ or tensorflow_ to make it clear
########################################################################################

# Get the length of the training dataset
len_train_ds = len(train_ds)
parallel_calls = tf.data.AUTOTUNE
cache_output_types = (tf.string, tf.float32, tf.int32, tf.string, tf.int32)
procs_output_types = (tf.float32, tf.float32, tf.int32, tf.string, tf.int32)

# Create the training dataset pipeline
train_dataset = (
    train_ds.shuffle(len_train_ds)
    .map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper, python_disk_cache_start, cache_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(
        functools.partial(
            python_fuction_wrapper, python_load_and_decode_file, procs_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper, python_audio_augmentations, procs_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(
        functools.partial(
            python_fuction_wrapper,
            python_dataset_melspectro_pipeline,
            procs_output_types,
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
    .map(tensorflow_image_augmentations, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper, python_disk_cache_end, procs_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
    .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
    .batch(SC["CLASSIFIER_BATCH_SIZE"])
    .prefetch(parallel_calls)
    .repeat(count=1)
)

# Create the validation dataset pipeline
validation_dataset = (
    val_ds.map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper, python_disk_cache_start, cache_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(
        functools.partial(
            python_fuction_wrapper, python_load_and_decode_file, procs_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper,
            python_dataset_melspectro_pipeline,
            procs_output_types,
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper, python_disk_cache_end, procs_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
    .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
    .batch(SC["CLASSIFIER_BATCH_SIZE"])
    .prefetch(parallel_calls)
    .repeat(count=1)
)

# Create the test dataset pipeline
test_dataset = (
    test_ds.map(tensorflow_add_variant_and_cache, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper, python_disk_cache_start, cache_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(
        functools.partial(
            python_fuction_wrapper, python_load_and_decode_file, procs_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_load_random_subsection, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper,
            python_dataset_melspectro_pipeline,
            procs_output_types,
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_reshape_image_pipeline, num_parallel_calls=parallel_calls)
    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
    .map(
        functools.partial(
            python_fuction_wrapper, python_disk_cache_end, procs_output_types
        ),
        num_parallel_calls=parallel_calls,
    )
    .map(tensorflow_output_shape_setter, num_parallel_calls=parallel_calls)
    .map(tensorflow_drop_variant_and_cache, num_parallel_calls=parallel_calls)
    .batch(SC["CLASSIFIER_BATCH_SIZE"])
    .prefetch(parallel_calls)
    .repeat(count=1)
)

########################################################################################
# Build and train the classification models
# Note: MODEL_NAME denotes the classification model used
########################################################################################

MODEL_NAME = "InceptionResNetV2"


def build_model(trainable):
    base_model = keras.applications.InceptionResNetV2(
        weights="imagenet",
        input_shape=(
            SC["MODEL_INPUT_IMAGE_HEIGHT"],
            SC["MODEL_INPUT_IMAGE_WIDTH"],
            SC["MODEL_INPUT_IMAGE_CHANNELS"],
        ),
        include_top=False,
    )

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(
        shape=(
            SC["MODEL_INPUT_IMAGE_HEIGHT"],
            SC["MODEL_INPUT_IMAGE_WIDTH"],
            SC["MODEL_INPUT_IMAGE_CHANNELS"],
        )
    )

    # Pre-trained weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    # x = scale_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(len(class_names))(x)
    model = keras.Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model


def train_model():
    if not os.path.exists(f"models/{MODEL_NAME}/"):
        os.makedirs(f"models/{MODEL_NAME}/", exist_ok=True)

    model = build_model(trainable=False)
    start_time = time.time()

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    # reduce learning rate to avoid overshooting local minima
    lr_reduce_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.75,
        patience=8,
        verbose=1,
        mode="min",
        cooldown=0,
        min_lr=1e-7,
    )

    # end the training if no improvement for 16 epochs in a row, then restore best model weights
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=16,
        verbose=0,
        mode="min",
        baseline=None,
        restore_best_weights=True,
    )

    # save the best model as it trains..
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        f"models/checkpoint_{MODEL_NAME}.hdf5",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        callbacks=[lr_reduce_plateau, early_stopping, mcp_save],
        epochs=SC["MAX_EPOCHS"],
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model Training Time: {elapsed_time} seconds")
    print(f"Model Training Time: {round(elapsed_time / 60, 2)} minutes")

    model.save(f"models/{MODEL_NAME}/", overwrite=True)
