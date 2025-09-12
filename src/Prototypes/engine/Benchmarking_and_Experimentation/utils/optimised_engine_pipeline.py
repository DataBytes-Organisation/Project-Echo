# utils/optimised_engine_pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, time, json
from typing import Tuple

import tensorflow as tf
import tensorflow_hub as hub

from config.system_config import SC
from config.model_configs import MODELS
from utils.data_pipeline import create_datasets
try:
    from utils.data_pipeline import build_datasets  # type: ignore
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

    from tensorflow.keras.applications import mobilenet_v3
    x = mobilenet_v3.preprocess_input(inputs)

    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=(h, w, c),
        pooling="avg"
    )
    backbone.trainable = True
    y = backbone(x, training=False)

    reg = tf.keras.regularizers.l2(l2_coefficient) if l2_regularization else None

    if use_classifier_head:
        y = tf.keras.layers.Dropout(dropout)(y)
        outputs = tf.keras.layers.Dense(
            num_classes, activation="softmax", kernel_regularizer=reg, name="classifier"
        )(y)
    else:
        outputs = y  

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{model_name}_clf")



def train_model(
    model_name: str,
    epochs: int = 2,
    batch_size: int = 16,
    l2_regularization: bool = False,
    l2_coefficient: float = 1e-4,
    use_classifier_head: bool = True,   
    dropout: float = 0.5              
):
    audio_dir = SC.get("AUDIO_DATA_DIRECTORY")
    if not audio_dir or not os.path.isdir(audio_dir):
        raise ValueError(f"Directory does not exist: {audio_dir}")

    SC["CLASSIFIER_BATCH_SIZE"] = batch_size

    ds_tuple = create_datasets(audio_dir)
    if len(ds_tuple) == 4:
        train_ds_init, val_ds_init, test_ds_init, class_names = ds_tuple
    else:
        train_ds_init, val_ds_init, class_names = ds_tuple
        test_ds_init = None

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

    lr = float(MODELS.get(model_name, {}).get("learning_rate", 1e-4))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(epochs),
        verbose=1,
    )

    return model, history

