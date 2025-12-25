#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
Make sure you have set the environment variable

    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

where hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx is your Huggingface access token.
"""

from typing import Any, Dict

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForCTC, AutoProcessor


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    model.metadata_props.clear()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
          x: (N, T, C), dtype float32
          mask: (N, T), dtype int64. Valid positions are 1. Padding positions are 0.
        Returns:
          logits: (N, T/4, vocob_size), dtype float32
          logits_len: (N,), dtype int64
        """
        o = self.m(x, mask.bool())
        logits_len = self.m._get_subsampling_output_length(mask.sum(-1)).to(torch.int64)
        return o.logits, logits_len


def generate_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    id2token = {i: t for t, i in vocab.items()}

    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(tokenizer.vocab_size):
            if i == tokenizer.pad_token_id:
                f.write(f"<blk> {i}\n")
            else:
                f.write(f"{id2token[i]} {i}\n")
    print("saved to tokens.txt")


@torch.no_grad()
def main():
    model_id = "google/medasr"
    processor = AutoProcessor.from_pretrained(model_id)

    generate_tokens(processor.tokenizer)

    model = AutoModelForCTC.from_pretrained(model_id)

    w = Wrapper(model)
    w.eval()

    filename = "model.onnx"
    x = torch.rand(1, 100, 128)
    mask = torch.ones(1, x.shape[1], dtype=torch.int64)
    torch.onnx.export(
        w,
        (x, mask),
        filename,
        input_names=["x", "mask"],
        output_names=["logits", "logits_len"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "mask": {0: "N", 1: "T"},
            "logits": {0: "N", 1: "T_4"},
            "logits_len": {0: "N"},
        },
        opset_version=14,
        #  external_data=False,
        #  dynamo=False,
    )

    meta_data = {
        "model_type": "medasr_ctc",
        "version": "20251225",
        "model_author": "google",
        "maintainer": "k2-fsa",
        "vocab_size": processor.tokenizer.vocab_size,
        "url": "https://github.com/Google-Health/medasr",
        "license": "https://developers.google.com/health-ai-developer-foundations/terms",
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
    main()
