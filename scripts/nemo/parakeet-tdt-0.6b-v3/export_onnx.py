#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from pathlib import Path
from typing import Dict
import os

import nemo.collections.asr as nemo_asr
import onnx
import onnxmltools
import torch
from onnxmltools.utils.float16_converter import (
    convert_float_to_float16,
    convert_float_to_float16_model_path,
)
from onnxruntime.quantization import QuantType, quantize_dynamic


def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(onnx_fp32_model, keep_io_types=True)
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


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
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v3"
    )

    asr_model.eval()

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    asr_model.encoder.export("encoder.onnx")
    asr_model.decoder.export("decoder.onnx")
    asr_model.joint.export("joiner.onnx")
    os.system("ls -lh *.onnx")

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "2",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        "comment": "Only the transducer branch is exported",
        "feat_dim": 128,
    }

    for m in ["encoder", "decoder", "joiner"]:
        quantize_dynamic(
            model_input=f"./{m}.onnx",
            model_output=f"./{m}.int8.onnx",
            weight_type=QuantType.QUInt8 if m == "encoder" else QuantType.QInt8,
        )
        os.system("ls -lh *.onnx")

        if m == "encoder":
            export_onnx_fp16_large_2gb(f"{m}.onnx", f"{m}.fp16.onnx")
        else:
            export_onnx_fp16(f"{m}.onnx", f"{m}.fp16.onnx")

    add_meta_data("encoder.int8.onnx", meta_data)
    add_meta_data("encoder.fp16.onnx", meta_data)
    print("meta_data", meta_data)


if __name__ == "__main__":
    main()
