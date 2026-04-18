import sys
import torch
import onnx
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
sys.path.append(MODEL_DIR)

from panns_cnn14 import PannsCNN14ArcFace

CHECKPOINT_PATH = r"\src\Prototypes\engine\torch_impl\model\cnn14\Cnn14_mAP=0.431.pth"
ONNX_OUTPUT_PATH = r"\src\Prototypes\engine\torch_impl\model\cnn14\cnn14.onnx"

DEVICE = "cpu"
NUM_CLASSES = 123

model = PannsCNN14ArcFace(
    classes_num=NUM_CLASSES,
    pretrained=False,
    use_arcface=False
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.eval()

dummy_input = torch.randn(1, 1, 64, 500, device=DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False,
)

m = onnx.load(ONNX_OUTPUT_PATH)
for x in m.opset_import:
    print("Opset:", x.version)