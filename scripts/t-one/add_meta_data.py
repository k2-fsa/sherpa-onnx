#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import onnx


def main():
    meta_data = {
        "model_type": "t-one",
        "language": "Russian",
        "version": 1,
        "maintainer": "k2-fsa",
        "sample_rate": 8000,
        "frame_length_ms": 300,  # chunk_duration_ms
        "state_dim": 219729,
        "comment": "This is a streaming CTC model for Russian with expected audio sample rate 8000",
        "url": "https://github.com/voicekit-team/T-one",
        "see_also": "https://huggingface.co/t-tech/T-one",
    }
    model = onnx.load("./model.onnx")

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    print("--------------------")

    print(model.metadata_props)

    onnx.save(model, "./model.onnx")


if __name__ == "__main__":
    main()
