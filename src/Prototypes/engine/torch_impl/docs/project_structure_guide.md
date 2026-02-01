# Project Structure and Development Guide

## Overview

This is a PyTorch-based audio classification framework/pipeline designed for training and deploying efficient neural networks on audio data.
The project supports multiple architectures, quantization-aware training (QAT), knowledge distillation, and various augmentation techniques.

## Project Structure

```
# Project Structure and Development Guide

## Overview

This is a PyTorch-based audio classification framework/pipeline designed for training and deploying efficient neural networks on audio data.
The project supports multiple architectures, quantization-aware training (QAT), knowledge distillation, and various augmentation techniques.

## Project Structure

```
torch-impl/
├── main.py                             # Main training/evaluation script
├── train.py                            # Training logic and utilities
├── dataset.py                          # Dataset handling and preprocessing
├── augment.py                          # Spectrogram augmentation (SpecAugment)
├── deploy_model.py                     # Exports trained models to TorchScript and TFLite
├── validate_models.py                  # Validates TFLite model against the PyTorch model
├── validating_audio_file.py            # Gradio UI for interactive audio validation
├── demo_iot.py                         # CLI demo for running inference with the TFLite model
├── pyproject.toml                      # Project dependencies and metadata
├── docs/                               # Documentation files
│   ├── download_data.md                # Instructions for downloading the background noise dataset
│   └── ...
├── config/                             # Configuration files (Hydra)
│   ├── config.yaml                     # Main configuration
│   ├── model/                          # Model-specific configs
│   │   ├── efficientnet_v2.yaml
│   │   ├── efficientnet_v2_qat.yaml
│   │   ├── ghost_efficientnet_v2.yaml
│   │   ├── panns_cnn14.yaml
│   │   ├── panns_mobilenetv1.yaml
│   │   └── panns_mobilenetv2.yaml
│   └── teacher_model/                  # Teacher model configs for distillation
│       └── [same as model/]
└── model/                              # Model architectures and utilities
    ├── __init__.py                     # Model factory and wrapper
    ├── effv2.py                        # EfficientNetV2 implementation
    ├── ghost_effv2.py                  # Custom Ghost EfficientNetV2
    ├── panns_cnn14.py                  # PANNs CNN14 architecture
    ├── panns_mobilenetv1.py            # PANNs MobileNetV1 architecture
    ├── panns_mobilenetv2.py            # PANNs MobileNetV2 architecture
    ├── quant.py                        # Quantization utilities
    └── utils.py                        # Utility functions and loss functions
```

## Core Components

### Data Pipeline (`dataset.py`)

- **SpectrogramDataset**: Main dataset class for audio processing
- **Features**:
  - Mel-spectrogram generation using librosa
  - Caching system with diskcache for performance (try to be as close as possible to the tensorflow pipeline)
  - Audio and spectrogram augmentations
  - Configurable audio preprocessing parameters

### 2. Model Factory (`model/__init__.py`)

- **Model**: Wrapper class that handles different architectures
- **Supported Models**:
  - EfficientNetV2 (with ArcFace support)
  - Custom Ghost EfficientNetV2
  - PANNs family (CNN14, MobileNetV1, MobileNetV2)
- **Features**:
  - Automatic model selection based on config
  - QAT (Quantization-Aware Training) support
  - Model fusion for inference optimization

### 3. Training Framework (`train.py`)

- **Trainer**: Comprehensive training class
- **Features**:
  - Mixed precision training with AMP
  - Support Gradient Clipping for more stable training
  - Early stopping and learning rate scheduling
  - TensorBoard logging
  - Knowledge distillation support
  - Checkpoint management

### 4. Configuration System

Uses Hydra for hierarchical configuration management:
- Modular configs for different components
- Easy experimentation with parameter sweeps
- Environment-specific overrides

## Data Setup

This project requires setting up your main dataset for training. Additionally, you can download the ESC-50 dataset which is used to provide background noise for data augmentation.

For instructions on downloading the background noise dataset, see [download_data.md](./download_data.md).

## Quick Start

### Basic Training

```bash
python main.py
```

### Override Configuration

```bash
python main.py model=efficientnet_v2 training.batch_size=32 training.epochs=50
```

### Test Only

```bash
python main.py run.train=false run.test=true run.checkpoint_path=path/to/model.pth
```

### Quantization

```bash
python main.py model=efficientnet_v2_qat run.quantise=true
```

## Development Guidelines

### Adding New Models

1. **Create Model Class** in `model/new_model.py`:

```python
class NewModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # Model implementation
            
    def forward(self, x, labels=None):
        # Forward pass
        return outputs
            
    def fuse_model(self):
        # Model fusion for quantization
        pass
```

2. **Register in Model Factory** (`model/__init__.py`):

```python
elif model_params.model_name == "new_model":
    self.model = NewModel(
        num_classes=model_params.num_classes,
        **model_params
    )
```

3. **Create Configuration** (`config/model/new_model.yaml`):

```yaml
name: new_model
params:
  num_classes: ${data.num_classes}
  # Additional parameters
```

### Adding New Augmentations

1. **Audio Augmentations**: Extend in `config.yaml` under `augmentations.audio`
2. **Spectrogram Augmentations**: Modify `augment.py` or add new transforms

### Custom Loss Functions

Add to `model/utils.py`:

```python
class CustomLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
            
    def forward(self, inputs, targets):
        # Loss computation
        return loss
```

## Configuration Management

### Key Configuration Sections

1. **System**: Data paths, caching settings
2. **Data**: Audio processing parameters, dataset splits
3. **Model**: Architecture selection and parameters
4. **Training**: Optimization, scheduling, augmentations
5. **Run**: Execution modes (train/test/quantize)

### Hydra Override Syntax

```bash
# Single parameter
python main.py training.learning_rate=0.001

# Multiple parameters
python main.py training.batch_size=64 training.epochs=100

# Nested parameters
python main.py model.params.pretrained=false

# List parameters
python main.py augmentations.audio.transforms[0].p=0.5
```

## Performance Optimization

### Data Loading

- Use `num_workers` appropriate for your system
- Enable `pin_memory=True` for GPU training
- Consider disk caching for repeated experiments

### Training

- Mixed precision training with AMP
- Gradient clipping for stability
- Learning rate scheduling

### Model Optimization

- Model fusion before quantization
- QAT for deployment optimization
- Knowledge distillation for model compression

## Debugging and Monitoring

### TensorBoard

Monitor training progress:

```bash
tensorboard --logdir outputs/
```

### Logging

The framework provides comprehensive logging:
- Training/validation losses
- Model parameters count
- GPU memory usage
- Checkpoint information

### Common Issues

1. **CUDA Memory**: Reduce batch size or enable gradient checkpointing
2. **Quantization Errors**: Ensure model fusion is properly implemented
3. **Audio Loading**: Check file formats and sample rates
4. **Configuration Conflicts**: Use `--cfg job` to debug Hydra configs

## Testing and Validation

### Model Validation

Use the built-in testing functionality:

```python
accuracy, precision, recall, f1 = trainer.test(test_loader)
```

## Deployment Considerations

### Model Export

For production deployment:

1. **Quantization**: Use QAT or post-training quantization
2. **ONNX Export**: Convert PyTorch models to ONNX format
3. **TorchScript**: Use JIT compilation for C++ deployment
4. **Convert to TFLite**: Use TFLite format for re-use previous code from other team

├── docs/                               # Documentation files
│   ├── download_data.md                # Instructions for downloading the background noise dataset
│   └── ...
├── config/                             # Configuration files (Hydra)
│   ├── config.yaml                     # Main configuration
│   ├── model/                          # Model-specific configs
│   │   ├── efficientnet_v2.yaml
│   │   ├── efficientnet_v2_qat.yaml
│   │   ├── ghost_efficientnet_v2.yaml
│   │   ├── panns_cnn14.yaml
│   │   ├── panns_mobilenetv1.yaml
│   │   └── panns_mobilenetv2.yaml
│   └── teacher_model/                  # Teacher model configs for distillation
│       └── [same as model/]
└── model/                              # Model architectures and utilities
    ├── __init__.py                     # Model factory and wrapper
    ├── effv2.py                        # EfficientNetV2 implementation
    ├── ghost_effv2.py                  # Custom Ghost EfficientNetV2
    ├── panns_cnn14.py                  # PANNs CNN14 architecture
    ├── panns_mobilenetv1.py            # PANNs MobileNetV1 architecture
    ├── panns_mobilenetv2.py            # PANNs MobileNetV2 architecture
    ├── quant.py                        # Quantization utilities
    └── utils.py                        # Utility functions and loss functions
```

## Core Components

### Data Pipeline (`dataset.py`)

- **SpectrogramDataset**: Main dataset class for audio processing
- **Features**:
  - Mel-spectrogram generation using librosa
  - Caching system with diskcache for performance (try to be as close as possible to the tensorflow pipeline)
  - Audio and spectrogram augmentations
  - Configurable audio preprocessing parameters

### 2. Model Factory (`model/__init__.py`)

- **Model**: Wrapper class that handles different architectures
- **Supported Models**:
  - EfficientNetV2 (with ArcFace support)
  - Custom Ghost EfficientNetV2
  - PANNs family (CNN14, MobileNetV1, MobileNetV2)
- **Features**:
  - Automatic model selection based on config
  - QAT (Quantization-Aware Training) support
  - Model fusion for inference optimization

### 3. Training Framework (`train.py`)

- **Trainer**: Comprehensive training class
- **Features**:
  - Mixed precision training with AMP
  - Support Gradient Clipping for more stable training
  - Early stopping and learning rate scheduling
  - TensorBoard logging
  - Knowledge distillation support
  - Checkpoint management

### 4. Configuration System

Uses Hydra for hierarchical configuration management:
- Modular configs for different components
- Easy experimentation with parameter sweeps
- Environment-specific overrides

## Data Setup

This project requires setting up your main dataset for training. Additionally, you can download the ESC-50 dataset which is used to provide background noise for data augmentation.

For instructions on downloading the background noise dataset, see [download_data.md](./download_data.md).

## Quick Start

### Basic Training

```bash
python main.py
```

### Override Configuration

```bash
python main.py model=efficientnet_v2 training.batch_size=32 training.epochs=50
```

### Test Only

```bash
python main.py run.train=false run.test=true run.checkpoint_path=path/to/model.pth
```

### Quantization

```bash
python main.py model=efficientnet_v2_qat run.quantise=true
```

## Development Guidelines

### Adding New Models

1. **Create Model Class** in `model/new_model.py`:

```python
class NewModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # Model implementation
            
    def forward(self, x, labels=None):
        # Forward pass
        return outputs
            
    def fuse_model(self):
        # Model fusion for quantization
        pass
```

2. **Register in Model Factory** (`model/__init__.py`):

```python
elif model_params.model_name == "new_model":
    self.model = NewModel(
        num_classes=model_params.num_classes,
        **model_params
    )
```

3. **Create Configuration** (`config/model/new_model.yaml`):

```yaml
name: new_model
params:
  num_classes: ${data.num_classes}
  # Additional parameters
```

### Adding New Augmentations

1. **Audio Augmentations**: Extend in `config.yaml` under `augmentations.audio`
2. **Spectrogram Augmentations**: Modify `augment.py` or add new transforms

### Custom Loss Functions

Add to `model/utils.py`:

```python
class CustomLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()
            
    def forward(self, inputs, targets):
        # Loss computation
        return loss
```

## Configuration Management

### Key Configuration Sections

1. **System**: Data paths, caching settings
2. **Data**: Audio processing parameters, dataset splits
3. **Model**: Architecture selection and parameters
4. **Training**: Optimization, scheduling, augmentations
5. **Run**: Execution modes (train/test/quantize)

### Hydra Override Syntax

```bash
# Single parameter
python main.py training.learning_rate=0.001

# Multiple parameters
python main.py training.batch_size=64 training.epochs=100

# Nested parameters
python main.py model.params.pretrained=false

# List parameters
python main.py augmentations.audio.transforms[0].p=0.5
```

## Performance Optimization

### Data Loading

- Use `num_workers` appropriate for your system
- Enable `pin_memory=True` for GPU training
- Consider disk caching for repeated experiments

### Training

- Mixed precision training with AMP
- Gradient clipping for stability
- Learning rate scheduling

### Model Optimization

- Model fusion before quantization
- QAT for deployment optimization
- Knowledge distillation for model compression

## Debugging and Monitoring

### TensorBoard

Monitor training progress:

```bash
tensorboard --logdir outputs/
```

### Logging

The framework provides comprehensive logging:
- Training/validation losses
- Model parameters count
- GPU memory usage
- Checkpoint information

### Common Issues

1. **CUDA Memory**: Reduce batch size or enable gradient checkpointing
2. **Quantization Errors**: Ensure model fusion is properly implemented
3. **Audio Loading**: Check file formats and sample rates
4. **Configuration Conflicts**: Use `--cfg job` to debug Hydra configs

## Testing and Validation

### Model Validation

Use the built-in testing functionality:

```python
accuracy, precision, recall, f1 = trainer.test(test_loader)
```

## Deployment Considerations

### Model Export

For production deployment:

1. **Quantization**: Use QAT or post-training quantization
2. **ONNX Export**: Convert PyTorch models to ONNX format
3. **TorchScript**: Use JIT compilation for C++ deployment
4. **Convert to TFLite**: Use TFLite format for re-use previous code from other team
