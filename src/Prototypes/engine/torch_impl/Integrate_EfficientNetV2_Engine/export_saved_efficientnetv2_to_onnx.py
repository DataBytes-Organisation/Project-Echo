"""
Export the saved Project Echo EfficientNetV2 model to ONNX.

This script uses the EfficientNetV2 model trained and saved in the current
Engine integration workflow.
"""

import pathlib

import torch
import timm
import onnx


BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "_trained_models"

MODEL_PATH = MODEL_DIR / "efficientnetv2_project_echo.pt"
ONNX_OUTPUT_PATH = MODEL_DIR / "efficientnetv2_project_echo.onnx"

DEVICE = "cpu"


def load_saved_model_for_export(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
        in_chans=checkpoint["in_chans"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


def export_to_onnx():
    print("Loading saved EfficientNetV2 model...")
    model, checkpoint = load_saved_model_for_export(MODEL_PATH)

    print("Model loaded successfully.")
    print("Model name:", checkpoint["model_name"])
    print("Number of classes:", checkpoint["num_classes"])

    # Shape must match our preprocessing:
    # batch, channel, n_mels, time_bins
    dummy_input = torch.randn(1, 1, 128, 313, device=DEVICE)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        dynamo=False,
    )

    print("ONNX model saved to:")
    print(ONNX_OUTPUT_PATH)

    print("Checking ONNX model...")
    onnx_model = onnx.load(ONNX_OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)

    for opset in onnx_model.opset_import:
        print("Opset:", opset.version)

    print("ONNX export completed successfully.")


if __name__ == "__main__":
    export_to_onnx()