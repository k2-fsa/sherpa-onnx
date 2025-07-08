#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import os
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


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

    asr_model.export(filename, onnx_opset_version=18)

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

    os.system("ls -lh *.onnx")

    quantize_dynamic(
        model_input="./model.onnx",
        model_output="./model.int8.onnx",
        weight_type=QuantType.QUInt8,
    )

    add_meta_data("model.int8.onnx", meta_data)

    os.system("ls -lh *.onnx")

    print("preprocessor", asr_model.cfg.preprocessor)
    print(meta_data)


if __name__ == "__main__":
    main()
