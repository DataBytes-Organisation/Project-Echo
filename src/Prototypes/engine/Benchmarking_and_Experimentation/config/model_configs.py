"""Configuration for different model architectures to benchmark."""

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

        "MobileNetV3-large": {
        "hub_url": "https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/classification/5",
        "trainable": True,
        "dense_layers": [8, 4],
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "expected_input_shape": (224, 224, 3),
    },

        "MobileNetV3-Small": {
        "hub_url": "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/classification/5",
        "trainable": True,
        "dense_layers": [8, 4],
        "dropout": 0.5,
        "learning_rate": 1e-4,
        "expected_input_shape": (224, 224, 3),
    },
}