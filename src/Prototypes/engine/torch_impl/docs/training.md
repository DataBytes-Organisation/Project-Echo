# Training Process Report

This document outlines the training and evaluation workflow, managed by the `Trainer` class (`train.py`) and configured via Hydra (`config/`).

## Overview

The training process is designed to be flexible and powerful, incorporating standard training loops with advanced techniques like metric learning, knowledge distillation, and mixed-precision training. The entire workflow is orchestrated by `main.py`, which reads configurations, prepares the data, initializes the model, and hands off control to the `Trainer`.

## Configuration-Driven Workflow

The project uses Hydra for configuration management. The main file, `config/config.yaml`, defines all aspects of a run.

**Key Configuration Sections:**

- system: Defines file paths for the audio dataset and disk cache.
    - data: Controls all parameters for audio processing and spectrogram generation (e.g., sample_rate, n_mels, hop_length).
    - training: Contains core hyperparameters for the training process.
        - `batch_size`, `epochs`, `learning_rate`
        - `optimizer` and `scheduler` are instantiated directly from this config.
        - `criterion`: The main loss function (e.g., `CrossEntropyLoss`).
        - `use_arcface`: Enables metric learning ("circle" or "arcface").
    - augmentations: Defines the pipeline for audio and spectrogram augmentations.
    - distillation: Configuration for knowledge distillation, including teacher model path, alpha, and temperature.

## Training Process (`train.py`)

The `Trainer` class encapsulates the entire training and evaluation logic.

### Training Loop (`_train_one_epoch`)

For each batch of data, the following steps are performed:

1. Data to Device: Inputs and labels are moved to the configured device (cuda or cpu).
2. Zero Gradients: The optimizer's gradients are reset.
3. Forward Pass: The student model processes the inputs to get logits. This happens within a torch.amp.autocast context for automatic mixed-precision training, which improves performance on modern GPUs.
4. Metric Learning (Optional): If enabled, the logits are passed through a metric learning module (CircleLoss or ArcMarginProduct) to compute modified logits that enforce an angular margin.
5. Loss Calculation:
    - Standard: The loss is calculated between the (potentially modified) logits and the ground-truth labels.
    - Distillation: If enabled, a composite loss is calculated: `(1 - alpha) * L_student + alpha * L_distill * T^2`. The distillation loss (`KLDivLoss`) measures the difference between the student's and teacher's softened probability distributions. The `T^2` term is used to scale the gradient from the distillation loss, ensuring its contribution is properly balanced with the student loss by using power of two of the distilation temperature.
6. Backpropagation: The loss is scaled by a GradScaler (to prevent underflow with mixed precision) and backpropagation is performed.
7. Gradient Clipping: To prevent exploding gradients, gradients are unscaled and then clipped to a maximum norm defined by `grad_clip`.
8. Optimizer Step: The optimizer updates the model's weights.
9. Scaler Update: The GradScaler updates its scale for the next iteration.

### Evaluation (`_evaluate`)

The evaluation loop runs on the validation set after each training epoch. It calculates the loss and accuracy without performing backpropagation (`torch.no_grad()`). The results are used for learning rate scheduling, early stopping, and identifying the best model checkpoint.

## Advanced Techniques

### Metric Learning

- Purpose: To train the model to produce more discriminative embeddings, where samples from the same class are closer together in the feature space and samples from different classes are farther apart.
- Implementation: Controlled by `training.use_arcface`. When set, the standard linear head is replaced with a `CosineLinear` layer, and the `CircleLoss` or `ArcMarginProduct` module is applied to the outputs before the final loss calculation.

### Knowledge Distillation

- Purpose: To transfer knowledge from a large, pre-trained "teacher" model to a smaller "student" model, improving the student's performance beyond what it could achieve alone.
- Implementation: Enabled via `training.distillation.enabled`. The Trainer loads the teacher model specified in `teacher_model_path` and uses it to generate "soft targets" for the student during training.

## Running an Experiment

### Starting a Run

Experiments are started from the command line via `main.py`. Hydra's command-line interface allows for easy modification of any configuration parameter.

```bash
# Run with default configuration from config.yaml

python main.py

# Override the model and number of epochs

python main.py model=efficientnet_v2_qat training.epochs=50

# Disable distillation

python main.py training.distillation.enabled=false
```

### Outputs and Artifacts
- Output Directory: Hydra automatically creates a unique output directory for each run (e.g., outputs/YYYY-MM-DD/HH-MM-SS/).
- Logs: All console output is saved to a .log file in this directory.
- Checkpoints: The best performing model checkpoint is saved as best_<model_name>.pth in the output directory.
- TensorBoard: Metrics for training and validation loss/accuracy are logged and can be visualized using TensorBoard: `tensorboard --logdir outputs/`
