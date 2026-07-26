# Ghost EfficientNetV2 Architecture Report

## Overview

**DISCLAIMER:** THIS ARCHITECTURE IS JUST WIP AND CURRENTLY NOT WORKING WELL.

The Ghost EfficientNetV2 is a custom lightweight architecture that combines the efficiency principles of EfficientNetV2 with GhostNet's parameter reduction techniques and ECA (Efficient Channel Attention) modules. This architecture is specifically designed for resource-constrained environments while maintaining competitive accuracy.

## Architecture Foundations

### GhostNet Principles

GhostNet addresses the redundancy in feature maps by generating "ghost" features through cheap linear operations instead of expensive convolutions.

#### Core Concept: Ghost Features

```
Traditional Conv: Input -> Conv(3x3) -> Output [All features computed independently]
Ghost Module: Input -> Primary Conv -> Cheap Operation -> Concatenate -> Output
```

**Key Insight**: Many feature maps in CNNs are similar to each other (ghost features). Instead of generating all features through expensive operations, generate a subset with regular convolutions and create the rest through cheap transformations.

#### Ghost Module Architecture

```python
class GhostModule(nn.Module):
def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1):
    # Primary convolution: generates init_channels = ceil(out_channels / ratio)
    self.primary_conv = nn.Conv2d(in_channels, init_channels, kernel_size, stride, bias=False)
    
    # Cheap operation: generates new_channels = init_channels * (ratio - 1) 
    # Depthwise conv
    self.cheap_operation = nn.Conv2d(
        init_channels,
        new_channels,
        dw_size,
        1,
        groups=init_channels,
        bias=False
    ) 
```

**Parameters Reduction**:
- Traditional Conv: `in_channels × out_channels × kernel_size²`  
- Ghost Module: `in_channels × init_channels × kernel_size² + init_channels × new_channels × dw_size²`
- Theoretical speedup: `ratio / (1 + 1/ratio) ≈ 2x` when ratio=2

### 2. ECA (Efficient Channel Attention)

ECA replaces the computationally expensive SE (Squeeze-and-Excitation) modules with a more efficient alternative.

#### SE vs ECA Comparison

**SE Module**:
```
Input -> GAP -> FC(reduce) -> ReLU -> FC(expand) -> Sigmoid -> Scale
Parameters: 2 × channels²/reduction_ratio
```

**ECA Module**:
```
Input -> GAP -> 1D Conv(k) -> Sigmoid -> Scale  
Parameters: k (typically 3-9)
```

#### ECA Implementation Details

```python
class ECA(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        # Adaptive kernel size based on channel dimension
        t = int(abs((math.log(in_channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)
```

**Adaptive Kernel Size**: `k = ψ(C) = |log₂(C)/γ + b/γ|odd`
- Larger channels -> Larger kernel size
- Captures long-range dependencies efficiently

## Ghost EfficientNetV2 Architecture

### Overall Structure

```
Stem -> Stage 1-6 -> Global Pool -> Classification Head
```

### Architecture Configuration

| Stage | Block Type | Expansion | Channels | Repeats | Stride |
|-------|------------|-----------|----------|---------|--------|
| Stem | Conv3x3 | - | 24 | 1 | 2 |
| 1 | FusedMBConv | 1 | 24 | 2 | 1 |
| 2 | FusedMBConv | 4 | 48 | 4 | 2 |
| 3 | FusedMBConv | 4 | 64 | 4 | 2 |
| 4 | MBConv | 4 | 128 | 6 | 2 |
| 5 | MBConv | 6 | 160 | 9 | 1 |
| 6 | MBConv | 6 | 256 | 15 | 2 |

### Building Blocks

#### 1. Ghost MBConv Block

```python
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, drop_rate):
        # Expansion with Ghost Module (if expand_ratio > 1)
        if expand_ratio != 1:
            layers.append(GhostModule(in_channels, hidden_dim, kernel_size=1))
        
        # Depthwise convolution  
        layers.append(ConvBnAct(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim))
        
        # ECA attention
        layers.append(ECA(hidden_dim))
        
        # Projection with Ghost Module
        layers.append(GhostModule(hidden_dim, out_channels, kernel_size=1, relu=False))
```

**Key Features**:
- Ghost modules replace expensive 1x1 convolutions
- ECA attention instead of SE
- Stochastic Depth for regularization
- LayerScale for training stability

#### 2. Ghost FusedMBConv Block

```python
class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, drop_rate):
        if expand_ratio != 1:
            # Fused expansion + depthwise with Ghost Module
            layers.append(GhostModule(in_channels, hidden_dim, kernel_size=3, stride=stride))
        else:
            # Regular depthwise when no expansion
            layers.append(ConvBnAct(in_channels, hidden_dim, kernel_size=3, groups=in_channels))
            
        layers.extend([ECA(hidden_dim), GhostModule(hidden_dim, out_channels, 1, relu=False)])
```

### Advanced Features

#### 1. Stochastic Depth

```python
class StochasticDepth(nn.Module):
    def stochastic_depth(self, x):
        if self.training:
            keep_prob = 1 - self.p
            random_tensor = keep_prob + torch.rand(shape, device=x.device)
            return x * random_tensor.floor().div_(keep_prob)
        return x
```

**Benefits**:
- Reduces overfitting
- Improves gradient flow
- Enables training of deeper networks

#### 2. LayerScale

```python
class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        self.gamma = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)))
    
    def forward(self, x):
        return x * self.gamma
```

**Purpose**:
- Stabilizes training of deep networks
- Allows better gradient flow in residual connections
- Improves convergence

## Performance Characteristics

### Model Scaling

The architecture supports width and depth multipliers:

```python
def __init__(self, width_mult=0.75, depth_mult=1.0, drop_rate=0.1):
    # Scale channel dimensions
    out_channels = int(24 * width_mult)
    
    # Scale layer repetitions  
    num_repeats = int(math.ceil(repeats * depth_mult))
```

### Efficiency Metrics

| Configuration | Parameters | Model Size |
|---------------|------------|------|
| width=0.5, depth=0.5 | 1.1M | 5.28M |
| width=0.75, depth=1.0 | 4.8M | 20.11M |
| width=1.0, depth=1.0 | 8.4M | 33.77M |

### Quantization Performance

The architecture is optimized for quantization:

```python
def fuse_model(self):
    # Fuse Conv-BN pairs in Ghost modules
    for m in self.modules():
        if hasattr(m, 'fuse'):
            m.fuse()
```

### Training Strategy

```yaml
# Recommended configuration
model:
  name: ghost_efficientnet_v2
  params:
    width_mult: 0.75        # Balance size/accuracy
    depth_mult: 1.0         # Full depth for audio complexity  
    drop_rate: 0.1          # Regularization
    use_arcface: true       # Improved feature learning
```

## Advantages and Trade-offs

### Advantages

1. **Parameter Efficiency**: 2-4x fewer parameters than EfficientNetV2
2. **Speed**: Faster inference, especially on mobile devices
3. **Memory**: Lower memory footprint during training/inference
4. **Quantization Friendly**: Designed for efficient quantization

### Trade-offs

1. **Accuracy**: Slight accuracy drop compared to full EfficientNetV2
2. **Training Complexity**: More components to tune (Ghost ratio, ECA parameters)
3. **Limited Pretrained Models**: No ImageNet pretraining available

## Use Cases

### Ideal Applications

- **Mobile/Edge deployment** with strict size constraints
- **Real-time audio processing** applications
- **Large-scale deployment** where computational cost matters
- **Embedded systems** with limited resources
