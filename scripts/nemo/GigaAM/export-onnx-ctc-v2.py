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
    model_name = "v2_ctc"
    model = gigaam.load_model(model_name, fp16_encoder=False, use_flash=False, download_root=".")
    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(model.cfg["labels"]):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")
    model.to_onnx(".")
    meta_data = {
        "vocab_size": len(model.cfg["labels"]) + 1,
        "normalize_type": "",
        "subsampling_factor": 4,
        "model_type": "EncDecCTCModel",
        "version": "1",
        "model_author": "https://github.com/salute-developers/GigaAM",
        "license": "https://github.com/salute-developers/GigaAM/blob/main/LICENSE",
        "language": "Russian",
        "is_giga_am": 1,
    }
    add_meta_data(f"./{model_name}.onnx", meta_data)
    quantize_dynamic(
        model_input=f"./{model_name}.onnx",
        model_output="./model.int8.onnx",
        weight_type=QuantType.QUInt8,
    )


if __name__ == '__main__':
    main()
