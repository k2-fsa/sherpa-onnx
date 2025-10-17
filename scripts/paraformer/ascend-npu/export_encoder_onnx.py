#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

from typing import List, Tuple

import torch
import yaml

from torch_model import Paraformer


def load_cmvn(filename) -> Tuple[List[float], List[float]]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = list(map(lambda x: float(x), t))
            else:
                inv_stddev = list(map(lambda x: float(x), t))

    return neg_mean, inv_stddev


def load_model():
    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("creating model")

    neg_mean, inv_stddev = load_cmvn("./am.mvn")

    neg_mean = torch.tensor(neg_mean, dtype=torch.float32)
    inv_stddev = torch.tensor(inv_stddev, dtype=torch.float32)

    m = Paraformer(
        neg_mean=neg_mean,
        inv_stddev=inv_stddev,
        input_size=560,
        vocab_size=8404,
        encoder_conf=config["encoder_conf"],
        decoder_conf=config["decoder_conf"],
        predictor_conf=config["predictor_conf"],
    )
    m.eval()

    print("loading state dict")
    state_dict = torch.load("./model_state_dict.pt", map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    m.load_state_dict(state_dict)
    del state_dict

    return m


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    x = torch.randn(1, 100, 560, dtype=torch.float32)

    opset_version = 14
    filename = "encoder.onnx"
    torch.onnx.export(
        model.encoder,
        x,
        filename,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["encoder_out"],
        dynamic_axes={
            "x": {1: "T"},
            "encoder_out": {1: "T"},
        },
    )

    print(f"Saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(20251013)
    main()
