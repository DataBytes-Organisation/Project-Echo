"""Define experiment combinations to run."""

from .augmentation_configs import AUDIO_AUGMENTATIONS, IMAGE_AUGMENTATIONS
from .model_configs import MODELS

# Define experiments to run
EXPERIMENTS = [
    {
        "name": "baseline",
        "audio_augmentation": "none",
        "image_augmentation": "none",
        "model": "EfficientNetV2B0",
        "epochs": 10,
        "batch_size": 16,
    },
    {
        "name": "basic_audio_aug",
        "audio_augmentation": "basic",
        "image_augmentation": "none",
        "model": "EfficientNetV2B0",
        "epochs": 10,
        "batch_size": 16,
    },
    {
        "name": "basic_image_aug",
        "audio_augmentation": "none",
        "image_augmentation": "basic_rotation",
        "model": "EfficientNetV2B0",
        "epochs": 10,
        "batch_size": 16,
    },
    {
        "name": "full_augmentation",
        "audio_augmentation": "advanced",
        "image_augmentation": "combined",
        "model": "EfficientNetV2B0",
        "epochs": 10,
        "batch_size": 16,
    },
    {
        "name": "mobilenet_baseline",
        "audio_augmentation": "none",
        "image_augmentation": "none",
        "model": "MobileNetV2",
        "epochs": 10,
        "batch_size": 16,
    },
    {
        "name": "mobilenet_full_aug",
        "audio_augmentation": "advanced",
        "image_augmentation": "combined",
        "model": "MobileNetV2",
        "epochs": 10,
        "batch_size": 16,
    }
]
    