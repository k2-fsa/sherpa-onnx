#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import onnx
import torch

from generate_voices_bin import speaker2id


def main():
    model = onnx.load("./kokoro.onnx")
    style = torch.load("./voices/zf_001.pt", weights_only=True, map_location="cpu")

    id2speaker_str = ""
    speaker2id_str = ""
    sep = ""
    for s, i in speaker2id.items():
        speaker2id_str += f"{sep}{s}->{i}"
        id2speaker_str += f"{sep}{i}->{s}"
        sep = ","

    meta_data = {
        "model_type": "kokoro",
        "language": "multi-lang, e.g., English, Chinese",
        "has_espeak": 1,
        "sample_rate": 24000,
        "version": 2,
        "voice": "en-us",
        "style_dim": ",".join(map(str, style.shape)),
        "n_speakers": len(speaker2id),
        "id2speaker": id2speaker_str,
        "speaker2id": speaker2id_str,
        "speaker_names": ",".join(map(str, speaker2id.keys())),
        "model_url": "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh",
        "maintainer": "k2-fsa",
        "comment": "This is Kokoro v1.1-zh, a multilingual TTS model, supporting English, Chinese.",
    }

    print(model.metadata_props)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    print("--------------------")

    print(model.metadata_props)

    onnx.save(model, "./kokoro.onnx")


if __name__ == "__main__":
    main()
