import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.ao.quantization as quant
import numpy as np

import torch
import torch.ao.quantization as quantization
from torch.ao.quantization.quantize_fx import (
	prepare_qat_fx as __prepare_qat_fx,
	prepare_fx,
	convert_fx,
)

default_qat_qconfig = quantization.get_default_qat_qconfig("fbgemm")

per_tensor_weight_handler = quantization.FakeQuantize.with_args(
	observer=quantization.MovingAverageMinMaxObserver,  # Per-tensor observer
	quant_min=-128,
	quant_max=127,
	dtype=torch.qint8,
	qscheme=torch.per_tensor_affine,
	reduce_range=False,
)

qconfig_per_tensor = quantization.QConfig(activation=default_qat_qconfig.activation, weight=per_tensor_weight_handler)

qconfig_mapping = (
	quantization.QConfigMapping()
	.set_object_type(torch.nn.Conv2d, qconfig_per_tensor)
	.set_object_type(torch.nn.Linear, qconfig_per_tensor)
	.set_object_type(torch.nn.BatchNorm2d, qconfig_per_tensor)
	.set_object_type(torch.nn.ReLU, qconfig_per_tensor)
	.set_object_type(torch.nn.ReLU6, qconfig_per_tensor)
)


def prepare_qat_fx(float_model, example_input):
    """
    Prepare a model for FX graph-mode quantisation-aware training
    using an example tensor from the real training pipeline.
    """
    if example_input is None:
        raise ValueError(
            "An example input tensor is required to prepare the model for QAT."
        )

    example_input = example_input.cpu()

    prepared_qat = __prepare_qat_fx(
        float_model,
        qconfig_mapping,
        example_inputs=(example_input,),
    )

    return prepared_qat


def prepare_post_static_quantize_fx(float_model, calib_dl):
    """
    Prepare a model for post-training static quantisation using
    an example tensor from the calibration DataLoader.
    """
    quant_model = copy.deepcopy(float_model).cpu().eval()

    try:
        example_inputs, _ = next(iter(calib_dl))
    except StopIteration:
        raise ValueError(
            "Calibration DataLoader is empty and cannot provide "
            "an example input for static quantisation."
        )

    example_input = example_inputs[:1].cpu()

    prepared = prepare_fx(
        quant_model,
        qconfig_mapping,
        example_inputs=(example_input,),
    )

    with torch.no_grad():
        for inputs, _ in calib_dl:
            prepared(inputs.cpu())

    return prepared
