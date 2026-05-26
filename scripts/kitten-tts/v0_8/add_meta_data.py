#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.

import argparse

import numpy as np
import onnx

from generate_voices_bin import speaker2id


NANO_SPEED_PRIORS = {
    "expr-voice-2-f": 0.8,
    "expr-voice-2-m": 0.8,
    "expr-voice-3-m": 0.8,
    "expr-voice-3-f": 0.8,
    "expr-voice-4-m": 0.9,
    "expr-voice-4-f": 0.8,
    "expr-voice-5-m": 0.8,
    "expr-voice-5-f": 0.8,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Input and output ONNX model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Hugging Face model name, e.g. KittenML/kitten-tts-mini-0.8",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(args.model)

    model = onnx.load(args.model)

    style = np.load("./voices.npz")
    style_shape = style[next(iter(style.keys()))].shape

    speaker2id_str = ""
    id2speaker_str = ""
    sep = ""
    for s, i in speaker2id.items():
        speaker2id_str += f"{sep}{s}->{i}"
        id2speaker_str += f"{sep}{i}->{s}"
        sep = ","

    if "kitten-tts-nano-0.8" in args.model_name:
        speed_priors = [NANO_SPEED_PRIORS[s] for s in speaker2id]
    else:
        speed_priors = [1.0] * len(speaker2id)

    meta_data = {
        "model_type": "kitten-tts",
        "language": "English",
        "has_espeak": 1,
        "sample_rate": 24000,
        "version": 8,
        "voice": "en-us",
        "max_token_len": 400,
        "start_id": 0,
        "end_id": 10,
        "pad_id": 0,
        "add_pad_after_end": 1,
        "style_dim": ",".join(map(str, style_shape)),
        "n_speakers": len(speaker2id),
        "speaker2id": speaker2id_str,
        "id2speaker": id2speaker_str,
        "speaker_names": ",".join(map(str, speaker2id.keys())),
        "speaker_speed_priors": ",".join(map(str, speed_priors)),
        "model_url": f"https://huggingface.co/{args.model_name}",
        "see_also": "https://github.com/KittenML/KittenTTS",
        "maintainer": "k2-fsa",
        "comment": f"This is {args.model_name} and supports only English",
    }

    print(model.metadata_props)

    del model.metadata_props[:]

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
