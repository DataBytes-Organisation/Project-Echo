[<- Back to Main Guide](model_integration_guide.md)

---

## Model Architectures

The project supports several neural network architectures, selectable via the configuration files.

### EfficientNetV2

-   **Description**: A state-of-the-art image classification model from Google, offering an excellent trade-off between accuracy and computational cost. It is adapted for this project to analyse audio spectrograms as images.
-   **Key Feature**: High accuracy, benefiting from pre-training on the large-scale ImageNet dataset, which provides a strong foundation for pattern recognition.
-   **Configuration (`config/model/efficientnet_v2.yaml`)**:
    -   `model_name`: `efficientnet_v2_s` selects the "small" model variant.
    -   `pretrained: true`: Loads the valuable weights learned from the ImageNet pre-training.
    -   `trainable_blocks`: Controls the fine-tuning strategy for the pre-trained layers.
        -   `0`: **Feature Extraction**. All pre-trained layers are frozen. Only the final, task-specific classification layer is trained. This is fast and suitable for smaller datasets.
        -   `N`: **Fine-Tuning**. The final layer and the last `N` blocks of the model are trained. This allows the model to adjust its learned features to the nuances of audio data.
        -   `-1`: **Full Training**. All model layers are unfrozen and trained. This is computationally intensive and best used with a large dataset.
    -   `replace_silu_with_relu`: (Optional `bool`, defaults to `false`) If `true`, all `SiLU` activation functions in the network are replaced with `ReLU`. This can be useful for hardware compatibility or performance tuning, but may impact accuracy.

### PANNs (Pre-trained Audio Neural Networks)

-   **Description**: A family of models specifically designed for audio classification and pre-trained on Google's AudioSet, a large-scale dataset of sound clips.
-   **Key Feature**: Strong out-of-the-box performance on audio tasks due to their domain-specific pre-training.
-   **Variations**:
    -   `panns_cnn14`: A 14-layer Convolutional Neural Network that serves as a robust baseline.
    -   `panns_mobilenetv1` / `panns_mobilenetv2`: Use the MobileNet architecture, known for its efficiency on resource-constrained devices.

### GhostEfficientNetV2 (Experimental)

-   A custom, experimental model developed for this project. It aims to be exceptionally parameter-efficient by combining techniques from `EfficientNetV2` and `GhostNet`. It is currently a work-in-progress.

---
[<- Back to Main Guide](model_integration_guide.md)
