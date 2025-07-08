#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import argparse
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import onnxmltools
import torch
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
from onnxruntime.quantization import QuantType, quantize_dynamic


def export_onnx_fp16_large_2gb(onnx_fp32_path, onnx_fp16_path):
    onnx_fp16_model = convert_float_to_float16_model_path(
        onnx_fp32_path, keep_io_types=True
    )
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


def add_meta_data(filename: str, meta_data: Dict[str, str]):
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
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt_ctc-0.6b-ja"
    )

    print(asr_model.cfg)
    print(asr_model)

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    decoder_type = "ctc"
    asr_model.change_decoding_strategy(decoder_type=decoder_type)
    asr_model.eval()

    asr_model.set_export_config({"decoder_type": "ctc"})

    filename = "model.onnx"

    asr_model.export(filename)

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "subsampling_factor": 8,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja",
        "comment": "Only the CTC branch is exported",
        "doc": "See https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja",
    }
    add_meta_data(filename, meta_data)

    export_onnx_fp16_large_2gb("model.onnx", "model.fp16.onnx")

    quantize_dynamic(
        model_input="./model.onnx",
        model_output="./model.int8.onnx",
        weight_type=QuantType.QUInt8,
    )

    print("preprocessor", asr_model.cfg.preprocessor)
    print(meta_data)


if __name__ == "__main__":
    main()
