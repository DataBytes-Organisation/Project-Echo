import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Configuration
CHECKPOINT_PATH = r".pth" # Path to your PyTorch model checkpoint
ONNX_OUTPUT_PATH = "EFF2_QAT_Circle_clean.onnx" # Output ONNX model path

# Model configuration
# Define your model parameters here
MODEL_NAME = "efficientnet_v2_s"
NUM_CLASSES = 118
USE_ARCFACE = True
IN_CHANNELS = 1
INPUT_SIZE = 224

# CosineLinear implementation for ArcFace
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input):
        # Normalize weights and input
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        normalized_input = F.normalize(input, p=2, dim=1)
        # Compute cosine similarity
        cosine = F.linear(normalized_input, normalized_weight)
        return cosine

# Import necessary components
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet, _efficientnet_conf, MBConv, FusedMBConv

MODEL_CONFIGS = {
    "efficientnet_v2_s": {"dropout": 0.2},
    "efficientnet_v2_m": {"dropout": 0.3},
    "efficientnet_v2_l": {"dropout": 0.4},
}

class EfficientNetV2ArcFace(EfficientNet):
    def __init__(
        self, model_name: str, pretrained: bool,
        num_classes: int, use_arcface: bool, in_channels: int = 1,
        trainable_blocks: int = 0
    ):
        if not model_name.startswith('efficientnet_v2'):
            raise ValueError("This class is for EfficientNetV2 models.")

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}. Try one of {list(MODEL_CONFIGS.keys())}")

        inverted_residual_setting, last_channel = _efficientnet_conf(model_name)
        dropout = MODEL_CONFIGS[model_name]["dropout"]
        original_weights_enum = getattr(models, f"EfficientNet_V2_{model_name.split('_')[-1].title()}_Weights")
        temp_num_classes = len(original_weights_enum.DEFAULT.meta["categories"]) if pretrained else num_classes
        super().__init__(inverted_residual_setting, dropout=dropout, last_channel=last_channel, num_classes=temp_num_classes)

        self.use_arcface = use_arcface
        self._modify_first_conv_layer(in_channels, pretrained)
        in_features = self.classifier[1].in_features
        if self.use_arcface:
            self.head = CosineLinear(in_features, num_classes)
        else:
            self.head = nn.Linear(in_features, num_classes)

        self.classifier = nn.Identity()
        self.set_trainable_layers(trainable_blocks)

    def _modify_first_conv_layer(self, in_channels: int, pretrained: bool):
        first_conv = self.features[0][0]
        original_in_channels = first_conv.in_channels
        if in_channels == original_in_channels: 
            return

        new_first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        if pretrained:
            original_weights = first_conv.weight.data
            if original_in_channels == 3 and in_channels == 1:
                new_weights = original_weights.mean(dim=1, keepdim=True) 
            else:
                new_weights = original_weights.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / original_in_channels
            new_first_conv.weight.data = new_weights

        self.features[0][0] = new_first_conv

    def set_trainable_layers(self, trainable_blocks: int = 0):
        for param in self.parameters(): param.requires_grad = False
        if trainable_blocks == -1:
            for param in self.parameters(): param.requires_grad = True
            return
        for param in self.head.parameters():
            param.requires_grad = True
        if trainable_blocks > 0:
            num_feature_blocks = len(self.features)
            trainable_blocks = min(trainable_blocks, num_feature_blocks)
            for i in range(num_feature_blocks - trainable_blocks, num_feature_blocks):
                for param in self.features[i].parameters(): param.requires_grad = True

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        embedding = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.head(embedding)

def load_model():
    """Load the PyTorch model from checkpoint"""
    print("=" * 60)
    print("Loading PyTorch Model")
    print("=" * 60)
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {MODEL_NAME}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Input Channels: {IN_CHANNELS}")
    print(f"  ArcFace: {USE_ARCFACE}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = EfficientNetV2ArcFace(
        model_name=MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
        use_arcface=USE_ARCFACE,
        in_channels=IN_CHANNELS,
        trainable_blocks=0
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  Loaded from 'state_dict' key")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  Loaded from 'model_state_dict' key")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"  Loaded from 'model' key")
        else:
            # Assume the entire dict is the state dict
            state_dict = checkpoint
            print(f"  Loaded entire checkpoint as state_dict")
        
        # Print additional info if available
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_acc' in checkpoint:
            print(f"  Best Accuracy: {checkpoint['best_acc']:.4f}")
        if 'val_acc' in checkpoint:
            print(f"  Validation Accuracy: {checkpoint['val_acc']:.4f}")
    else:
        state_dict = checkpoint
        print(f"  Loaded checkpoint directly")
    
    # Load state dict into model
    # For QAT models, we need to filter out quantization parameters
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    # Find keys that don't match
    extra_keys = checkpoint_keys - model_keys
    missing_keys = model_keys - checkpoint_keys
    
    if extra_keys:
        print(f"\n  Found {len(extra_keys)} extra keys (likely QAT params) - filtering...")
        # Filter out QAT-specific keys
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if not any(x in k for x in ['fake_quant', 'observer', 'activation_post_process'])}
        state_dict = filtered_state_dict
    
    if missing_keys:
        print(f"  Missing {len(missing_keys)} keys in checkpoint")
        for key in list(missing_keys)[:5]:
            print(f"    - {key}")
    
    # Try loading with strict=False to handle QAT models
    result = model.load_state_dict(state_dict, strict=False)
    
    if result.missing_keys:
        print(f"\n  Warning: {len(result.missing_keys)} keys not loaded")
        for key in result.missing_keys[:5]:
            print(f"    - {key}")
    if result.unexpected_keys:
        print(f"\n  Info: {len(result.unexpected_keys)} unexpected keys (QAT params)")
    
    model.eval()
    
    print("Model loaded successfully")
    return model

def verify_model(model):
    """Run a test inference to verify the model works"""
    print("\n" + "=" * 60)
    print("Verifying Model with Test Input")
    print("=" * 60)
    
    # Create dummy input
    dummy_input = torch.randn(1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    
    print(f"\nTest input shape: {dummy_input.shape}")
    print(f"Test input range: [{dummy_input.min():.6f}, {dummy_input.max():.6f}]")
    
    # Run inference
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"Output mean: {output.mean():.6f}, std: {output.std():.6f}")
    
    # Check for invalid values
    if torch.isnan(output).any():
        print("\n ERROR: Model output contains NaN values!")
        return False
    if torch.isinf(output).any():
        print("\n ERROR: Model output contains Inf values!")
        return False
    if output.abs().max() > 1e6:
        print(f"\n WARNING: Model output values are very large (max: {output.abs().max():.2e})")
        print("  This might indicate an issue with the model")
        return False
    
    print("\n Model verification passed - outputs look reasonable")
    return True

def export_to_onnx(model):
    """Export PyTorch model to ONNX format"""
    print("\n" + "=" * 60)
    print("Exporting to ONNX")
    print("=" * 60)
    
    # Create dummy input
    dummy_input = torch.randn(1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    
    # Export to ONNX
    print(f"\nExporting to: {ONNX_OUTPUT_PATH}")
    print("This may take a minute...")
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=13,  # Use opset 13 for better compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✓ ONNX model saved to: {ONNX_OUTPUT_PATH}")
    
    # Check file size
    file_size = os.path.getsize(ONNX_OUTPUT_PATH) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")

def verify_onnx_model(model):
    """Verify the exported ONNX model produces correct outputs"""
    print("\n" + "=" * 60)
    print("Verifying ONNX Model")
    print("=" * 60)
    
    import onnxruntime as ort
    import numpy as np
    
    # Load ONNX model
    print(f"\nLoading ONNX model: {ONNX_OUTPUT_PATH}")
    session = ort.InferenceSession(ONNX_OUTPUT_PATH)
    
    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    
    # Run ONNX inference
    print("Running ONNX inference...")
    onnx_output = session.run(None, {'input': test_input})[0]
    
    print(f"\nONNX output shape: {onnx_output.shape}")
    print(f"ONNX output range: [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")
    print(f"ONNX output mean: {onnx_output.mean():.6f}, std: {onnx_output.std():.6f}")
    
    # Run PyTorch inference with same input
    print("\nComparing with PyTorch output...")
    model.eval()
    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(test_input)).numpy()
    
    print(f"PyTorch output range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
    print(f"PyTorch output mean: {pytorch_output.mean():.6f}, std: {pytorch_output.std():.6f}")
    
    # Compare outputs
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\nComparison:")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")
    print(f"  Relative error: {(max_diff / (np.abs(pytorch_output).max() + 1e-10)):.8f}")
    
    if max_diff < 1e-5:
        print("\n ONNX export PASSED - outputs match PyTorch exactly!")
        return True
    elif max_diff < 1e-3:
        print("\n ONNX export PASSED - outputs match PyTorch within tolerance")
        return True
    else:
        print("\n ONNX export FAILED - outputs differ significantly from PyTorch")
        return False

def main():
    print("=" * 60)
    print("CLEAN PYTORCH TO ONNX CONVERTER")
    print("=" * 60)
    
    try:
        # Load model
        model = load_model()
        
        # Verify model works
        if not verify_model(model):
            print("\n✗ Model verification failed. Please check your checkpoint.")
            return
        
        # Export to ONNX
        export_to_onnx(model)
        
        # Verify ONNX model
        if verify_onnx_model(model):
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
            print(f"\n Clean ONNX model created: {ONNX_OUTPUT_PATH}")
            print("\nNext steps:")
            print("1. Run the TFLite conversion: python convert_onnx_to_tflite.py")
            print("3. Run validation: python validate_models.py")
        else:
            print("\n ONNX verification failed. The model may have issues.")
            
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()