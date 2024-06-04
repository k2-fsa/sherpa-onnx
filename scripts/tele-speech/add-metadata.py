#!/usr/bin/env python3

import json
from typing import Dict

import onnx
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
        meta.value = value

    onnx.save(model, filename)


def main():
    with open("./vocab.json", "r", encoding="utf-8") as f:
        tokens = json.load(f)

    vocab_size = len(tokens)
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for token, idx in tokens.items():
            if idx == 0:
                f.write("<blk> 0\n")
            else:
                f.write(f"{token} {idx}\n")

    filename = "model.onnx"
    meta_data = {
        "model_type": "telespeech_ctc",
        "version": "1",
        "model_author": "Tele-AI",
        "comment": "See also https://github.com/lovemefan/telespeech-asr-python",
        "license": "https://github.com/Tele-AI/TeleSpeech-ASR/blob/master/TeleSpeech%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf",
        "url": "https://github.com/Tele-AI/TeleSpeech-ASR",
    }

    add_meta_data(filename, meta_data)

    filename_int8 = f"model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )

    #  filename_uint8 = f"model.uint8.onnx"
    #  quantize_dynamic(
    #      model_input=filename,
    #      model_output=filename_uint8,
    #      op_types_to_quantize=["MatMul"],
    #      weight_type=QuantType.QUInt8,
    #  )


if __name__ == "__main__":
    main()
