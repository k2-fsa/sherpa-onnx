#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import argparse

import onnx
import torch

from generate_voices_bin import speaker2id


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="input and output onnx model"
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args.model)

    model = onnx.load(args.model)

    style = torch.load(
        "./kLegacy/v0.19/voices/af.pt", weights_only=True, map_location="cpu"
    )

    speaker2id_str = ""
    id2speaker_str = ""
    sep = ""
    for s, i in speaker2id.items():
        speaker2id_str += f"{sep}{s}->{i}"
        id2speaker_str += f"{sep}{i}->{s}"
        sep = ","

    meta_data = {
        "model_type": "kokoro",
        "language": "English",
        "has_espeak": 1,
        "sample_rate": 24000,
        "version": 1,
        "voice": "en-us",
        "style_dim": ",".join(map(str, style.shape)),
        "n_speakers": len(speaker2id),
        "speaker2id": speaker2id_str,
        "id2speaker": id2speaker_str,
        "speaker_names": ",".join(map(str, speaker2id.keys())),
        "model_url": "https://huggingface.co/hexgrad/kLegacy/",
        "see_also": "https://huggingface.co/spaces/hexgrad/Kokoro-TTS",
        "maintainer": "k2-fsa",
        "comment": "This is kokoro v0.19 and supports only English",
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

    onnx.save(model, args.model)

    print(f"Please see {args.model}, ./voices.bin, and ./tokens.txt")


if __name__ == "__main__":
    main()
