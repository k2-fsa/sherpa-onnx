#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
from typing import Dict

import onnx
import torch
from fairseq2.nn.batch_layout import BatchLayout
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from onnxruntime.quantization import QuantType, quantize_dynamic


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-card",
        type=str,
        required=True,
        help="omniASR_CTC_300M, or omniASR_CTC_1B",
    )
    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, str], model_card: str):
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

    if "300M" in model_card:
        onnx.save(model, filename)
    else:
        external_filename = filename.split(".onnx")[0]
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_filename + ".weights",
        )


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Args:
          x: (N, num_samples), float32
        """
        batch_layout = BatchLayout(shape=x.shape, seq_lens=[x.shape[1]])
        logits, _ = self.model(x, batch_layout)
        return logits


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))
    pipeline = ASRInferencePipeline(
        model_card=args.model_card,
        device="cpu",
        dtype=torch.float32,
    )

    vocab_size = pipeline.tokenizer._model.vocabulary_size

    with open("tokens.txt", "w") as f:
        for i in range(pipeline.tokenizer._model.vocabulary_size):
            f.write(f"{pipeline.tokenizer._model.index_to_token(i)} {i}\n")

    print("saved to tokens.txt")

    wrapper = ModelWrapper(pipeline.model)
    wrapper.eval()

    x = torch.rand(1, 16000 * 10)
    torch.onnx.export(
        wrapper,
        x,
        "model.onnx",
        opset_version=14,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "N", 1: "num_samples"},
            "logits": {0: "N", 1: "num_frames"},
        },
    )

    meta_data = {
        "vocab_size": vocab_size,
        "model_type": "omnilingual-asr",
        "version": "1",
        "sample_rate": 16000,
        "model_author": "facebookresearch",
        "url": "https://github.com/facebookresearch/omnilingual-asr",
        "comment": "300M-CTC",
    }

    add_meta_data("model.onnx", meta_data, args.model_card)
    print("saved to model.onnx")

    quantize_dynamic(
        model_input="./model.onnx",
        model_output="./model.int8.onnx",
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QUInt8,
    )
    print("saved to model.int8.onnx")


if __name__ == "__main__":
    main()
