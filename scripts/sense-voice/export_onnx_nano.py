#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import os
from typing import Any, Dict, List, Tuple

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from test_nano_torch import load_tokens, load_torch_model


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
    )
    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))
    id2tokens = load_tokens()

    vocab_size = len(id2tokens)
    blank_id = vocab_size - 1

    print("loading model")

    model = load_torch_model()
    model.eval()

    x = torch.randn(1, 30, 560, dtype=torch.float32)

    opset_version = args.opset_version
    filename = "model.onnx"
    torch.onnx.export(
        model,
        x,
        filename,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={
            "x": {1: "T"},
        },
    )

    model_author = "FunAudioLLM"
    comment = os.environ.get("comment", "FunAudioLLM/Fun-ASR-Nano-2512")
    url = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512"

    meta_data = {
        "lfr_window_size": 7,
        "lfr_window_shift": 6,
        "normalize_samples": 0,  # input should be in the range [-32768, 32767]
        "model_type": "sense_voice_ctc",
        "version": "1",
        "model_author": model_author,
        "maintainer": "k2-fsa",
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "comment": comment,
        "url": url,
    }
    add_meta_data(filename=filename, meta_data=meta_data)

    filename_int8 = "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        # Note that we have to use QUInt8 here.
        #
        # When QInt8 is used, C++ onnxruntime produces incorrect results
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    torch.manual_seed(20251217)
    main()
