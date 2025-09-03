import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import os
from pathlib import Path
import librosa
import matplotlib.pyplot as plt
import warnings
import sys
import time
import datetime


warnings.filterwarnings("ignore")

# Add the parent directory so the config module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.system_config import SC  # Import system settings
from config.model_configs import MODELS  # Import model configurations
from utils.data_pipeline import create_datasets, build_datasets   # Import dataset creation function

# For brevity, create a dummy dataset. # Actual data loading has now been implemented. This is a depreciated function.
def create_dummy_dataset(num_samples=100, num_classes=3):
    images = tf.random.uniform(
        (num_samples, SC['MODEL_INPUT_IMAGE_HEIGHT'], SC['MODEL_INPUT_IMAGE_WIDTH'], SC['MODEL_INPUT_IMAGE_CHANNELS'])
    )
    labels = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(SC['CLASSIFIER_BATCH_SIZE'])
    return dataset


def build_model(model_name="EfficientNetV2B0", num_classes=3, l2_regularization=False, l2_coefficient=0.001):

    """Builds a TensorFlow Keras model with optional L2 regularization.

    This function constructs a Keras Sequential model for image classification, leveraging
    pre-trained feature extractors from TensorFlow Hub. It allows for customization of the
    model architecture, including the choice of feature extractor, the number of dense layers,
    dropout rate, and the application of L2 regularization.

    Args:
        model_name (str, optional): Name of the model architecture to use. This name must
            correspond to a key in the `MODELS` dictionary defined in `config/model_configs.py`.
            Defaults to "EfficientNetV2B0".
        num_classes (int, optional): Number of output classes for the classification task.
            Defaults to 3.
        l2_regularization (bool, optional): Whether to apply L2 regularization to the dense layers.
            Defaults to False.
        l2_coefficient (float, optional): The L2 regularization coefficient. This value is used
            only if `l2_regularization` is True. Defaults to 0.001.

    Returns:
        tf.keras.Model: A compiled TensorFlow Keras model.

    Raises:
        ValueError: If the input dimensions specified in the system configuration (`SC`) do not
            match the expected input dimensions for the selected model, as defined in the
            `MODELS` dictionary. This ensures that the data pipeline handles image resizing
            appropriately.
    """
    
    config = MODELS.get(model_name, {})
    print(f"Model configuration: {config}")
    hub_url = config.get("hub_url", "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2")
    trainable = config.get("trainable", True)
    dense_layers = config.get("dense_layers", [8, 4])
    dropout = config.get("dropout", 0.5)

    # Get the expected input shape for this model from config; default to system config if not provided.
    expected_input_shape = config.get("expected_input_shape", (
        SC['MODEL_INPUT_IMAGE_HEIGHT'],
        SC['MODEL_INPUT_IMAGE_WIDTH'],
        SC['MODEL_INPUT_IMAGE_CHANNELS']
    ))
    expected_height, expected_width, expected_channels = expected_input_shape

    print(f"Building model: {model_name} with expected input shape: {expected_input_shape}")
    layers = [
        tf.keras.layers.InputLayer(input_shape=(
            expected_height,
            expected_width,
            expected_channels
        ))
    ]
    
    # Check if the input dimensions match the expected dimensions
    # if the input dimensions do not match, raise an error 
    # I think this is not needed as the data pipeline should resize the images to the expected dimensions.   
    # if (SC['MODEL_INPUT_IMAGE_HEIGHT'], SC['MODEL_INPUT_IMAGE_WIDTH']) != (expected_height, expected_width):
    #    raise ValueError(f"System config input dimensions ({SC['MODEL_INPUT_IMAGE_HEIGHT']}, {SC['MODEL_INPUT_IMAGE_WIDTH']}) do not match expected input dimensions ({expected_height}, {expected_width}).  The data pipeline must resize the images.")
        
    # Add normalization layer
    layers.extend([
        hub.KerasLayer(hub_url, trainable=trainable),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
    ])

    reg = regularizers.l2(l2_coefficient) if l2_regularization else None

    for mult in dense_layers:
        layers.append(tf.keras.layers.Dense(num_classes * mult, activation="relu", kernel_regularizer=reg))
        layers.append(tf.keras.layers.BatchNormalization())

    layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(num_classes, activation=None, kernel_regularizer=reg))
    
    model = tf.keras.Sequential(layers)
    return model



def train_model(model_name="EfficientNetV2B0", epochs=None, batch_size=None, l2_regularization=False, l2_coefficient=0.001):
    """Trains a specified model using datasets created by the data pipeline.

    Args:
        model_name (str, optional): Name of the model architecture to use. Defaults to "EfficientNetV2B0".
        epochs (int, optional): Number of training epochs. Overrides SC['MAX_EPOCHS'] if provided. Defaults to None.
        batch_size (int, optional): Batch size for training. Overrides SC['CLASSIFIER_BATCH_SIZE'] if provided. Defaults to None.
        l2_regularization (bool, optional): Whether to apply L2 regularization. Defaults to False.
        l2_coefficient (float, optional): The L2 regularization coefficient. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the trained model and the training history.
            - model (tf.keras.Model): The trained TensorFlow Keras model.
            - history (tf.keras.callbacks.History): The training history object.
    """
    # Check if the model name is valid
    if model_name not in MODELS:
        raise ValueError(f"Model name '{model_name}' not found in model_configs.py. Available models are: {list(MODELS.keys())}")

    if epochs is not None:
        SC['MAX_EPOCHS'] = epochs

    if batch_size is not None:
        SC['CLASSIFIER_BATCH_SIZE'] = batch_size

    # Create initial datasets (paths and labels)
    train_ds_init, val_ds_init, test_ds_init, class_names = create_datasets(SC['AUDIO_DATA_DIRECTORY'])
    num_classes = len(class_names)

    # Build the full data pipelines using build_datasets from data_pipeline.py
    train_dataset, validation_dataset, test_dataset = build_datasets(
        train_ds_init, val_ds_init, test_ds_init, class_names, model_name=model_name
    )

    # Build the model
    model = build_model(
        model_name,
        num_classes=num_classes,
        l2_regularization=l2_regularization,
        l2_coefficient=l2_coefficient
    )

    # Set the learning rate from the model config
    learning_rate = MODELS.get(model_name, {}).get("learning_rate", 1e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Define callbacks based on the notebook example
    log_dir = os.path.join("tensorboard_logs", model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    lr_reduce_plateau = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.75,
        patience=8,
        verbose=1,
        mode='min',
        min_lr=1e-7
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=16, # Increased patience as per notebook example
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    # Ensure models directory exists
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, f"checkpoint_{model_name}.hdf5")
    mcp_save = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    callbacks = [lr_reduce_plateau, early_stopping, tensorboard_callback, mcp_save]

    # Start training timer
    start_time = time.time()

    print(f"Starting training for model: {model_name}")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=SC['MAX_EPOCHS'],
        callbacks=callbacks
    )

    # End training timer and print duration
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Model training finished for {model_name}. Duration: {training_time:.2f} seconds")

    import json

    # After training is complete and class_names is available
    
    class_names_path = os.path.join(models_dir, f"class_names_{model_name}.json")
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved to {class_names_path}")

    return model, history

if __name__ == "__main__":
    # Example usage
    # Set the model name and parameters as needed
    # Example: Train Model for 10 epochs with batch size 16 and L2 regularization
    trained_model, training_history = train_model(
        "MobileNetV2",
        epochs=10,
        batch_size=16,
        l2_regularization=True,
        l2_coefficient=0.001
    )


