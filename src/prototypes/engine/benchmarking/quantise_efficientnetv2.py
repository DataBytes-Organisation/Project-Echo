"""
Generate the TFLite quantisation sweep for the production EfficientNetV2 model.

Pipeline (matches the archived production conversion path in
`.delete/archive/.../Integrate_EfficientNetV2_Engine/`):

    PyTorch .pt -> ONNX -> TF SavedModel -> TFLite {fp32, float16, dynamic-range, int8}

This script covers the last step: SavedModel -> TFLite variants, using
TensorFlow's own converter (per-channel int8 by default). It produces five
variants for the fp32-vs-int8/float16 calibration + size + latency comparison.

Producing the SavedModel from the ONNX is a separate, one-off step done with
onnx2tf (or the repo's onnx-tf script). It needs a modern TensorFlow (>=2.15),
so run it in a throwaway env, NOT the pinned TF 2.10 engine venv:

    onnx2tf -i efficientnetv2_project_echo.onnx -o effnet_savedmodel -nuo

The fp32 TFLite produced here was verified numerically identical to the shipped
`efficientnetv2_project_echo.tflite` (max logit diff ~1e-5), so the variants are
faithfully "the production model, quantised".

Usage:
    python quantise_efficientnetv2.py \
        --saved-model effnet_savedmodel \
        --calib-npy calib_nhwc.npy \
        --out-dir results_efficientnet/variants
"""

import argparse
import os

import numpy as np
import tensorflow as tf


VARIANT_ORDER = ["float32", "float16", "dynamic_range", "int8", "full_int8"]


def _representative_dataset(calib_nhwc):
    """Yield calibration samples one at a time, shaped (1, H, W, C)."""

    def gen():
        for i in range(len(calib_nhwc)):
            yield [calib_nhwc[i : i + 1]]

    return gen


def build_variant(saved_model_dir, variant, rep_gen=None):
    """Return the serialised TFLite bytes for a single quantisation variant."""
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if variant == "float32":
        pass
    elif variant == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif variant == "dynamic_range":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == "int8":
        # int8 weights + activations, float fallback I/O (per-channel default)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_gen
    elif variant == "full_int8":
        # full integer, int8 in/out (for edge/NPU targets)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    else:
        raise ValueError(f"unknown variant: {variant}")

    return conv.convert()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved-model", required=True,
                    help="TF SavedModel dir (produced from the ONNX via onnx2tf)")
    ap.add_argument("--calib-npy", required=True,
                    help="representative dataset as an (N, H, W, C) float32 .npy "
                         "of preprocessed clips (NHWC), used for int8 calibration")
    ap.add_argument("--out-dir", default="results_efficientnet/variants")
    ap.add_argument("--variants", nargs="+", default=VARIANT_ORDER)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    calib = np.load(args.calib_npy).astype(np.float32)
    rep_gen = _representative_dataset(calib)

    print(f"SavedModel: {args.saved_model}")
    print(f"Calibration set: {calib.shape}\n")
    for variant in args.variants:
        data = build_variant(args.saved_model, variant, rep_gen)
        out = os.path.join(args.out_dir, f"efficientnetv2_project_echo_{variant}.tflite")
        with open(out, "wb") as f:
            f.write(data)
        print(f"  {variant:14s} {len(data) / 1048576:6.1f} MB  ->  {out}")

    print("\nDone. Score them with benchmark_quant_variants.py.")


if __name__ == "__main__":
    main()
