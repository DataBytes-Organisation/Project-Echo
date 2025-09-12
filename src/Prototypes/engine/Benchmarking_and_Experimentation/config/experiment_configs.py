"""
Define experiment combinations to run. Manually specify desired experiments
with chosen audio and image augmentation strategies.
"""

from .augmentation_configs import AUDIO_AUGMENTATIONS, IMAGE_AUGMENTATIONS
from .model_configs import MODELS

# Define experiments to run
EXPERIMENTS = [
    # {
    #     "name": "baseline",
    #     "audio_augmentation": "none",
    #     "image_augmentation": "none",
    #     "model": "EfficientNetV2B0",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    # {
    #     "name": "noise_and_stretch_audio_aug",
    #     "audio_augmentation": "noise_and_stretch",
    #     "image_augmentation": "none",
    #     "model": "EfficientNetV2B0",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    # {
    #     "name": "basic_image_aug",
    #     "audio_augmentation": "none",
    #     "image_augmentation": "basic_rotation",
    #     "model": "EfficientNetV2B0",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    # {
    #     "name": "full_augmentation",
    #     "audio_augmentation": "advanced",
    #     "image_augmentation": "combined",
    #     "model": "EfficientNetV2B0",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    # {
    #     "name": "mobilenet_baseline",
    #     "audio_augmentation": "none",
    #     "image_augmentation": "none",
    #     "model": "MobileNetV2",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    # {
    #     "name": "mobilenet_full_aug",
    #     "audio_augmentation": "advanced",
    #     "image_augmentation": "combined",
    #     "model": "MobileNetV2",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    # {
    #     "name": "noise_and_pitch_small",
    #     "audio_augmentation": "noise_and_pitch",
    #     "image_augmentation": "none",
    #     "model": "ResNet50V2",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    # {
    #     "name": "spec_only_aggressive",
    #     "audio_augmentation": "none",
    #     "image_augmentation": "aggressive",
    #     "model": "InceptionV3",
    #     "epochs": 10,
    #     "batch_size": 16,
    # },
    #yz
    {
    "name": "mobilenetv3small_baseline",
    "audio_augmentation": "none",
    "image_augmentation": "none",
    "model": "MobileNetV3Small_224",
    "epochs": 2,
    "batch_size": 16
    },
    {
        "name": "efficientnetlite0_baseline",
        "audio_augmentation": "none",
        "image_augmentation": "none",
        "model": "EfficientNetLite0_224",
        "epochs": 2,
        "batch_size": 16
    },
]

    