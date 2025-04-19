import os

import gigaam
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic


def add_meta_data(filename: str, meta_data: dict[str, str]):
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


def main() -> None:
    model_name = "v2_rnnt"
    model = gigaam.load_model(
        model_name, fp16_encoder=False, use_flash=False, download_root="."
    )
    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(model.cfg["labels"]):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")
    model.to_onnx(".")
    meta_data = {
        "vocab_size": model.cfg["head"]["decoder"]["num_classes"],
        "pred_rnn_layers": model.cfg["head"]["decoder"]["pred_rnn_layers"],
        "pred_hidden": model.cfg["head"]["decoder"]["pred_hidden"],
        "normalize_type": "",
        "subsampling_factor": 4,
        "model_type": "EncDecRNNTBPEModel",
        "version": "1",
        "model_author": "https://github.com/salute-developers/GigaAM",
        "license": "https://github.com/salute-developers/GigaAM/blob/main/LICENSE",
        "language": "Russian",
        "is_giga_am": 1,
    }

    add_meta_data(f"./{model_name}_encoder.onnx", meta_data)
    quantize_dynamic(
        model_input=f"./{model_name}_encoder.onnx",
        model_output="./encoder.int8.onnx",
        weight_type=QuantType.QUInt8,
    )
    os.rename(f"./{model_name}_decoder.onnx", "decoder.onnx")
    os.rename(f"./{model_name}_joint.onnx", "joiner.onnx")
    os.remove(f"./{model_name}_encoder.onnx")


if __name__ == "__main__":
    main()
