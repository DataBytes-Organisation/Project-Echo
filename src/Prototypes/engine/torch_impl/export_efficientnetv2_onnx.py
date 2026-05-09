import timm
import torch
import onnx

ONNX_OUTPUT_PATH = r"\src\Prototypes\engine\torch_impl\model\efficientnetv2_rw_s\efficientnetv2_rw_s.onnx"

DEVICE = "cpu"
NUM_CLASSES = 123  

model = timm.create_model(
    "efficientnetv2_rw_s",
    pretrained=True,
    num_classes=NUM_CLASSES,
    in_chans=1,
).to(DEVICE)

model.eval()

dummy_input = torch.randn(1, 1, 128, 313, device=DEVICE)

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