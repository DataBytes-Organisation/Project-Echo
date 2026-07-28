"""Configuration for different model architectures to benchmark."""

from config.system_config import SC  # import global settings

MODELS = {
    "EfficientNetV2B0": {
        "hub_url": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2",
        "trainable": True,
        "dense_layers": [8, 4],
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "expected_input_shape": (260, 260, 3),
    },
    "MobileNetV2": {
        "hub_url": "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
        "trainable": True,
        "dense_layers": [8, 4],
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "expected_input_shape": (224, 224, 3),
    },
    "ResNet50V2": {
        "hub_url": "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5",
        "trainable": True,
        "dense_layers": [8, 4],
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "expected_input_shape": (224, 224, 3),
    },
    "InceptionV3": {
        "hub_url": "https://tfhub.dev/google/imagenet/inception_v3/classification/5",
        "trainable": True,
        "dense_layers": [8, 4],
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "expected_input_shape": (299, 299, 3),
    },
    "MobileNetV2_TF": {
        "constructor": "utils.models.mobilenetv2_tf.build_model",
        "params": {
            "num_classes": SC["NUM_CLASSES"],
            "input_height": SC["MODEL_INPUT_IMAGE_HEIGHT"],
            "input_width": SC["MODEL_INPUT_IMAGE_WIDTH"],
            "input_channels": SC["MODEL_INPUT_IMAGE_CHANNELS"],
            "dropout": 0.5,
            "lr": 1e-4,
            "base_trainable": False,
            "fine_tune_at": None,
        },
        "expected_input_shape": (
            SC["MODEL_INPUT_IMAGE_HEIGHT"],
            SC["MODEL_INPUT_IMAGE_WIDTH"],
            SC["MODEL_INPUT_IMAGE_CHANNELS"],
        ),
    },
}
