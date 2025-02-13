#!/usr/bin/env python3

from typing import Any, Dict

import onnx


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


def main():
    meta_data = {
        "model_type": "matcha-tts",
        "language": "Persian+English",
        "voice": "fa",
        "has_espeak": 1,
        "jieba": 0,
        "n_speakers": 1,
        "sample_rate": 22050,
        "version": 1,
        "pad_id": 0,
        "use_icefall": 0,
        "model_author": "Ali Mahmoudi (@mah92)",
        "maintainer": "k2-fsa",
        "use_eos_bos": 0,
        "num_ode_steps": 5,
        "see_also": "https://github.com/k2-fsa/sherpa-onnx/issues/1779",
    }
    add_meta_data("./female/model.onnx", meta_data)
    add_meta_data("./male/model.onnx", meta_data)


if __name__ == "__main__":
    main()
