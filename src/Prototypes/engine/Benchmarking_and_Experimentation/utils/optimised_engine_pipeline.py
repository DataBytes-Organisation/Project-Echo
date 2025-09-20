# utils/optimised_engine_pipeline.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, json
from typing import Tuple
import tensorflow as tf

from config.system_config import SC
from config.model_configs import MODELS
from utils.data_pipeline import create_datasets
try:
    from utils.data_pipeline import build_datasets
    _HAS_BUILD_PIPELINES = True
except Exception:
    _HAS_BUILD_PIPELINES = False


def build_model(
    model_name: str,
    num_classes: int,
    l2_regularization: bool = False,
    l2_coefficient: float = 1e-4,
    use_classifier_head: bool = True,
    dropout: float = 0.5
) -> tf.keras.Model:
    h = int(SC.get("MODEL_INPUT_IMAGE_HEIGHT", 224))
    w = int(SC.get("MODEL_INPUT_IMAGE_WIDTH", 224))
    c = int(SC.get("MODEL_INPUT_IMAGE_CHANNELS", 3))
    inputs = tf.keras.Input(shape=(h, w, c), dtype=tf.float32, name="input")

    preprocess_map = {
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "MobileNetV3Small": tf.keras.applications.mobilenet_v3.preprocess_input,
        "MobileNetV3Large": tf.keras.applications.mobilenet_v3.preprocess_input,
        "EfficientNetV2B0": tf.keras.applications.efficientnet_v2.preprocess_input,
        "ResNet50V2": tf.keras.applications.resnet_v2.preprocess_input,
        "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
    }
    preprocess_fn = preprocess_map.get(model_name, lambda x: x)
    x = preprocess_fn(inputs)

    backbone_map = {
        "MobileNetV2": tf.keras.applications.MobileNetV2,
        "MobileNetV3Small": tf.keras.applications.MobileNetV3Small,
        "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
        "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
        "ResNet50V2": tf.keras.applications.ResNet50V2,
        "InceptionV3": tf.keras.applications.InceptionV3,
    }
    backbone_class = backbone_map.get(model_name)
    if backbone_class is None:
        raise ValueError(f"Unsupported model: {model_name}")

 # For brevity, create a dummy dataset. # Actual data loading has now been implemented. This is a deprecated function.
def create_dummy_dataset(num_samples=100, num_classes=3):
    images = tf.random.uniform(
        (num_samples, SC['MODEL_INPUT_IMAGE_HEIGHT'], SC['MODEL_INPUT_IMAGE_WIDTH'], SC['MODEL_INPUT_IMAGE_CHANNELS'])
    )
    backbone.trainable = True
    y = backbone(x, training=False)


def build_model(model_name="EfficientNetV2B0", num_classes=3, l2_regularization=False, l2_coefficient=0.001):

    """Builds a TensorFlow Keras model with optional L2 regularisation.

    This function constructs a Keras Sequential model for image classification, using
    pre-trained feature extractors from TensorFlow Hub. It allows for customisation of the
    model architecture, including the choice of feature extractor, the number of dense layers,
    dropout rate, and the application of L2 regularisation.

    Args:
        model_name (str, optional): Name of the model architecture to use. This name must
            correspond to a key in the `MODELS` dictionary defined in `config/model_configs.py`.
            Defaults to "EfficientNetV2B0".
        num_classes (int, optional): Number of output classes for the classification task.
            Defaults to 3.
        l2_regularization (bool, optional): Whether to apply L2 regularisation to the dense layers.
            Defaults to False.
        l2_coefficient (float, optional): The L2 regularisation coefficient. This value is used
            only if `l2_regularization` is True. Defaults to 0.001.

    Returns:
        tf.keras.Model: A compiled TensorFlow Keras model.

    Raises:
        ValueError: If the input dimensions specified in the system configuration (`SC`) do not
            match the expected input dimensions for the selected model, as defined in the
            `MODELS` dictionary. This ensures that the data pipeline handles image resizing
            appropriately.
    """
    
    reg = tf.keras.regularizers.l2(l2_coefficient) if l2_regularization else None
    if use_classifier_head:
        y = tf.keras.layers.Dropout(dropout, name="head_dropout")(y)
        outputs = tf.keras.layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=reg,
            name="classifier"
        )(y)
    else:
        outputs = y

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
        
    # Add normalisation layer
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


def train_model(
    model_name: str,
    epochs: int = 2,
    batch_size: int = 16,
    l2_regularization: bool = False,
    l2_coefficient: float = 1e-4,
    use_classifier_head: bool = True,
    dropout: float = 0.5,
    use_callbacks: bool = True
):
    audio_dir = SC.get("AUDIO_DATA_DIRECTORY")
    if not audio_dir or not os.path.isdir(audio_dir):
        raise ValueError(f"Directory does not exist: {audio_dir}")

    SC["CLASSIFIER_BATCH_SIZE"] = batch_size

    Args:
        model_name (str, optional): Name of the model architecture to use. Defaults to "EfficientNetV2B0".
        epochs (int, optional): Number of training epochs. Overrides SC['MAX_EPOCHS'] if provided. Defaults to None.
        batch_size (int, optional): Batch size for training. Overrides SC['CLASSIFIER_BATCH_SIZE'] if provided. Defaults to None.
        l2_regularization (bool, optional): Whether to apply L2 regularisation. Defaults to False.
        l2_coefficient (float, optional): The L2 regularisation coefficient. Defaults to 0.001.

    Returns:
        tuple: A tuple containing the trained model and the training history.
            - model (tf.keras.Model): The trained TensorFlow Keras model.
            - history (tf.keras.callbacks.History): The training history object.
    
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

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        l2_regularization=l2_regularization,
        l2_coefficient=l2_coefficient,
        use_classifier_head=use_classifier_head,
        dropout=dropout
    )

    if _HAS_BUILD_PIPELINES:
        train_ds, val_ds, test_ds = build_datasets(
            train_ds_init, val_ds_init, test_ds_init, class_names, model_name=model_name
        )
    else:
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds_init.batch(batch_size).prefetch(AUTOTUNE)
        val_ds = val_ds_init.batch(batch_size).prefetch(AUTOTUNE)
        test_ds = None if test_ds_init is None else test_ds_init.batch(batch_size).prefetch(AUTOTUNE)

    loss_fn = "categorical_crossentropy"
    sample_batch = next(iter(train_ds.take(1)))
    if len(sample_batch[1].shape) == 1:
        loss_fn = "sparse_categorical_crossentropy"

    lr = float(MODELS.get(model_name, {}).get("learning_rate", 1e-4))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    callbacks = []
    if use_callbacks:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(SC.get("OUTPUT_DIRECTORY", "."), "logs"))
        ]

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
        patience=32,  # Allow for multiple LR reductions (4x lr patience)
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    # Ensure models directory exists
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create timestamp for unique model naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"checkpoint_{model_name}_{timestamp}.hdf5"
    checkpoint_path = os.path.join(models_dir, checkpoint_filename)
    
    mcp_save = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,  # Save only the best model
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    callbacks = [
        lr_reduce_plateau, early_stopping, tensorboard_callback, mcp_save
    ]

    # Start training timer
    start_time = time.time()

    print(f"Starting training for model: {model_name}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        verbose=1,
        callbacks=callbacks
    )

    out_dir = SC.get("OUTPUT_DIRECTORY", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"class_names_{model_name}.json"), "w") as f:
        json.dump(class_names, f)

    return model, history
