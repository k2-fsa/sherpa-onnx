#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import torch

from export_encoder_onnx import load_model
from torch_model import CifPredictorV2

if __name__ == "__main__":

    def modified_predictor_forward(self: CifPredictorV2, hidden: torch.Tensor):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(
            alphas * self.smooth_factor - self.noise_threshold
        )

        alphas = alphas.squeeze(-1)

        return alphas

    CifPredictorV2.forward = modified_predictor_forward


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    x = torch.randn(1, 100, 512, dtype=torch.float32)

    opset_version = 14
    filename = "predictor.onnx"
    torch.onnx.export(
        model.predictor,
        x,
        filename,
        opset_version=opset_version,
        input_names=["encoder_out"],
        output_names=["alphas"],
        dynamic_axes={
            "encoder_out": {1: "T"},
            "alphas": {1: "T"},
            },
    )
    print(f"Saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
