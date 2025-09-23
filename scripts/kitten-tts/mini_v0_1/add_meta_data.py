#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import argparse

import numpy as np
import onnx

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

    style = np.load("./voices.npz")
    style_shape = style[list(style.keys())[0]].shape

    speaker2id_str = ""
    id2speaker_str = ""
    sep = ""
    for s, i in speaker2id.items():
        speaker2id_str += f"{sep}{s}->{i}"
        id2speaker_str += f"{sep}{i}->{s}"
        sep = ","

    meta_data = {
        "model_type": "kitten-tts",
        "language": "English",
        "has_espeak": 1,
        "sample_rate": 24000,
        "version": 1,
        "voice": "en-us",
        "style_dim": ",".join(map(str, style_shape)),
        "n_speakers": len(speaker2id),
        "speaker2id": speaker2id_str,
        "id2speaker": id2speaker_str,
        "speaker_names": ",".join(map(str, speaker2id.keys())),
        "model_url": "https://huggingface.co/KittenML/kitten-tts-nano-0.2",
        "see_also": "https://github.com/KittenML/KittenTTS",
        "maintainer": "k2-fsa",
        "comment": "This is kitten-tts-nano-0.2 and supports only English",
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

    print(f"Please see {args.model}")


if __name__ == "__main__":
    main()
