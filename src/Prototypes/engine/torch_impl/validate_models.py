import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import os

from torchvision.models.efficientnet import EfficientNet, _efficientnet_conf
import torchvision.models as models

# Model paths
PYTORCH_MODEL_PATH = r".pth" # Your existing PyTorch model
TFLITE_MODEL_PATH = "EFF2_QAT_Circle_clean.tflite"  # Your existing TFLite model

# Model configuration
# You can modify these parameters as needed
MODEL_NAME = "efficientnet_v2_s"
NUM_CLASSES = 118
USE_ARCFACE = True
IN_CHANNELS = 1
INPUT_SIZE = 224

MODEL_CONFIGS = {
    "efficientnet_v2_s": {"dropout": 0.2},
    "efficientnet_v2_m": {"dropout": 0.3},
    "efficientnet_v2_l": {"dropout": 0.4},
}

# CosineLinear implementation for ArcFace
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input):
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        normalized_input = F.normalize(input, p=2, dim=1)
        cosine = F.linear(normalized_input, normalized_weight)
        return cosine

class EfficientNetV2ArcFace(EfficientNet):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int, 
                 use_arcface: bool, in_channels: int = 1, trainable_blocks: int = 0):
        if not model_name.startswith('efficientnet_v2'):
            raise ValueError("This class is for EfficientNetV2 models.")
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")

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
            in_channels=in_channels, out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size, stride=first_conv.stride,
            padding=first_conv.padding, bias=first_conv.bias is not None
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
        for param in self.parameters(): 
            param.requires_grad = False
        if trainable_blocks == -1:
            for param in self.parameters(): 
                param.requires_grad = True
            return
        for param in self.head.parameters():
            param.requires_grad = True
        if trainable_blocks > 0:
            num_feature_blocks = len(self.features)
            trainable_blocks = min(trainable_blocks, num_feature_blocks)
            for i in range(num_feature_blocks - trainable_blocks, num_feature_blocks):
                for param in self.features[i].parameters(): 
                    param.requires_grad = True

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        embedding = torch.flatten(self.avgpool(self.features(x)), 1)
        return self.head(embedding)

def load_pytorch_model():
    """Load PyTorch model"""
    print("Loading PyTorch model...")
    
    model = EfficientNetV2ArcFace(
        model_name=MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES,
        use_arcface=USE_ARCFACE, in_channels=IN_CHANNELS, trainable_blocks=0
    )
    
    checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Filter out QAT-specific keys
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if not any(x in k for x in ['fake_quant', 'observer', 'activation_post_process'])}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    
    print("PyTorch model loaded")
    return model

def run_pytorch_inference(model, input_data):
    """Run PyTorch inference"""
    with torch.no_grad():
        torch_input = torch.from_numpy(input_data)
        output = model(torch_input)
        output_np = output.numpy() if isinstance(output, torch.Tensor) else output
    return output_np

def run_tflite_inference(input_data):
    """Run TFLite inference"""
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Check if we need to transpose (NCHW -> NHWC)
    if len(input_details[0]['shape']) == 4 and input_details[0]['shape'][3] == IN_CHANNELS:
        tflite_input = np.transpose(input_data, (0, 2, 3, 1))
    else:
        tflite_input = input_data
    
    interpreter.set_tensor(input_details[0]['index'], tflite_input.astype(input_details[0]['dtype']))
    interpreter.invoke()
    output_np = interpreter.get_tensor(output_details[0]['index'])
    
    return output_np

def compare_models(num_tests=10):
    """Compare PyTorch and TFLite models across multiple test inputs"""
    print("=" * 70)
    print("PyTorch vs TFLite Model Validation")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Classes: {NUM_CLASSES}")
    print(f"  Input: {IN_CHANNELS}x{INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  ArcFace: {USE_ARCFACE}")
    print(f"  Number of tests: {num_tests}")
    
    # Load PyTorch model
    pytorch_model = load_pytorch_model()
    
    # Check TFLite model exists
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"\n TFLite model not found at: {TFLITE_MODEL_PATH}")
        return
    
    print(f" TFLite model found\n")
    
    print("=" * 70)
    print("Running comparisons...")
    print("=" * 70)
    
    all_diffs = []
    all_relative_errors = []
    
    for test_num in range(num_tests):
        # Create test input
        np.random.seed(42 + test_num)
        test_input = np.random.randn(1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        
        # Run both models
        pytorch_output = run_pytorch_inference(pytorch_model, test_input)
        tflite_output = run_tflite_inference(test_input)
        
        # Reshape if needed
        if pytorch_output.shape != tflite_output.shape:
            if np.prod(pytorch_output.shape) == np.prod(tflite_output.shape):
                tflite_output = tflite_output.reshape(pytorch_output.shape)
        
        # Calculate differences
        diff = np.abs(pytorch_output - tflite_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        relative_error = max_diff / (np.abs(pytorch_output).max() + 1e-10)
        
        all_diffs.append(max_diff)
        all_relative_errors.append(relative_error)
        
        # Get top predictions
        pytorch_top = np.argsort(pytorch_output[0])[-3:][::-1]
        tflite_top = np.argsort(tflite_output[0])[-3:][::-1]
        
        predictions_match = np.array_equal(pytorch_top, tflite_top)
        
        print(f"\nTest {test_num + 1}:")
        print(f"  PyTorch top-3: {pytorch_top} → values: {pytorch_output[0][pytorch_top]}")
        print(f"  TFLite  top-3: {tflite_top} → values: {tflite_output[0][tflite_top]}")
        print(f"  Predictions match: {'✓' if predictions_match else '✗'}")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        print(f"  Relative error: {relative_error:.8f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    avg_max_diff = np.mean(all_diffs)
    max_max_diff = np.max(all_diffs)
    avg_relative_error = np.mean(all_relative_errors)
    
    print(f"\nAcross {num_tests} tests:")
    print(f"  Average max difference: {avg_max_diff:.8f}")
    print(f"  Worst max difference: {max_max_diff:.8f}")
    print(f"  Average relative error: {avg_relative_error:.8f}")
    
    tolerance = 1e-3
    if max_max_diff < tolerance:
        print(f"\n ALL TESTS PASSED (within tolerance {tolerance})")
        print("  The TFLite model produces equivalent outputs to PyTorch!")
    elif max_max_diff < 0.01:
        print(f"\n MINOR DIFFERENCES DETECTED (max: {max_max_diff:.6f})")
        print("  Differences are small and likely acceptable for most use cases.")
    else:
        print(f"\n SIGNIFICANT DIFFERENCES DETECTED (max: {max_max_diff:.6f})")
        print("  The models may not be equivalent. Review the conversion process.")
    
    print("=" * 70)

if __name__ == "__main__":
    compare_models(num_tests=10)