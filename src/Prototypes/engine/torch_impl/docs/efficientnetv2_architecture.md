# EfficientNetV2 Architecture Report

## Overview

EfficientNetV2 is an advanced neural network architecture that improves upon the original EfficientNet by addressing training bottlenecks and enhancing training speed while maintaining excellent accuracy-efficiency trade-offs.

## Architecture Fundamentals

### Core Design Principles

1. **Compound Scaling**: Simultaneously scales network depth, width, and resolution
2. **Progressive Learning**: Gradually increases image size and regularization during training
3. **Improved Building Blocks**: Uses both MBConv and FusedMBConv blocks
4. **Neural Architecture Search (NAS)**: Optimized architecture discovered through automated search

### Key Improvements Over EfficientNetV1

- **Fused-MBConv blocks** for early layers (better training speed)
- **Smaller expansion ratios** to reduce memory access cost
- **Smaller kernel sizes** (3x3) to compensate for reduced depth
- **Progressive training** strategy for better convergence

## Architecture Details

### Model Variants

| Model | Parameters | 
|-------|------------|
| EfficientNetV2-S | 21.5M |
| EfficientNetV2-M | 54.1M |
| EfficientNetV2-L | 119.5M |

### Building Blocks

#### MBConv (Mobile Inverted Bottleneck Convolution)

```
Input -> Expand (1x1 Conv) -> Depthwise (3x3 Conv) -> Squeeze-Excite -> Project (1x1 Conv) -> Output
```

**Key Features**:
- Inverted residual structure
- Depthwise separable convolutions
- Squeeze-and-Excite attention
- Skip connections when input/output dimensions match

#### FusedMBConv (Fused Mobile Inverted Bottleneck)

```
Input -> Expand + Depthwise (3x3 Conv) -> Squeeze-Excite -> Project (1x1 Conv) -> Output
```

**Advantages**:
- Reduces memory access cost
- Better acceleration on modern hardware
- Used primarily in early layers

### Architecture Configuration

#### EfficientNetV2-S (Used in Project)

| Stage | Operator | Stride | Channels | Layers | Expansion |
|-------|----------|--------|----------|--------|-----------|
| 0 | Conv3x3 | 2 | 24 | 1 | - |
| 1 | FusedMBConv | 1 | 24 | 2 | 1 |
| 2 | FusedMBConv | 2 | 48 | 4 | 4 |
| 3 | FusedMBConv | 2 | 64 | 4 | 4 |
| 4 | MBConv | 2 | 128 | 6 | 4 |
| 5 | MBConv | 1 | 160 | 9 | 6 |
| 6 | MBConv | 2 | 256 | 15 | 6 |
| 7 | Conv1x1 + Pooling | - | 1280 | 1 | - |

## Implementation Details (Project-Specific)

### Custom Adaptations

#### Input Channel Modification

```python
def _modify_first_conv_layer(self, in_channels: int, pretrained: bool):
    # Adapts first conv layer from 3 channels (RGB) to 1 channel (grayscale spectrograms)
    if original_in_channels == 3 and in_channels == 1:
        new_weights = original_weights.mean(dim=1, keepdim=True)
```

**Purpose**: Convert RGB pretrained weights to grayscale spectrogram input

#### ArcFace/CircleLoss Integration

```python
if self.use_arcface:
    self.head = CosineLinear(in_features, num_classes)
else:
    self.head = nn.Linear(in_features, num_classes)
```

**Benefits**:
- Improved feature discriminability
- Better performance on classification tasks
- Angular margin penalty for better separation

#### Layer Freezing Strategy

```python
def set_trainable_layers(self, trainable_blocks: int = 0):
    # Freeze backbone, only train classification head
    # Optionally unfreeze last N blocks for fine-tuning
```

**Strategies**:
- `trainable_blocks = 0`: Only train classification head
- `trainable_blocks = N`: Train head + last N feature blocks
- `trainable_blocks = -1`: Train entire network

### Training Optimizations

#### Model Fusion for Quantization

```python
def fuse_model(self):
    # Fuse Conv2d + BatchNorm2d layers
    for module in self.modules():
        if isinstance(module, (MBConv, FusedMBConv)):
            self._fuse_conv_bn(module.block)
```

**Benefits**:
- Reduced memory footprint
- Faster inference
- Prerequisite for quantization

## Performance Characteristics

### Computational Efficiency

```
Model Summary: EfficientNetV2ArcFace
-----------------------------------------------------------------
Total Parameters          : 20,328,072
Trainable Parameters      : 20,328,048
Non-trainable Parameters  : 24
Estimated Model Size (MB) : 79.60
```

## Configuration Example

```yaml
# config/model/efficientnet_v2.yaml
name: efficientnet_v2
params:
  model_name: "efficientnet_v2_s"
  pretrained: true
  num_classes: 6
  use_arcface: true
  trainable_blocks: -1
```
