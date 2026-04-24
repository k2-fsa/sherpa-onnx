#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json
from typing import Any, Dict

import onnx
from iso639 import Lang

# For the following model,
# https://huggingface.co/rhasspy/piper-voices/blob/main/zh/zh_CN/xiao_ya/medium/zh_CN-xiao_ya-medium.onnx.json
# it uses g2pw, not espeak-ng.
# To handle that, we use has_g2pw = 1 in the meta_data


def get_args():
    # For en_GB-semaine-medium
    # --name semaine
    # --kind medium
    # --lang en_GB
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--kind",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=True,
    )
    return parser.parse_args()


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


def load_config(filename):
    with open(filename, "r") as file:
        config = json.load(file)
    return config


def generate_tokens(config):
    id_map = config["phoneme_id_map"]
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for s, i in id_map.items():
            if s == "\n":
                continue
            if isinstance(i, list):
                i = i[0]
            print(f"{s} {i}")
            f.write(f"{s} {i}\n")
    print("Generated tokens.txt")


# for en_US-lessac-medium.onnx
# export LANG=en_US
# export TYPE=lessac
# export NAME=medium
def main():
    args = get_args()
    print(args)
    lang = args.lang

    lang_iso = Lang(lang.split("_")[0])
    print(lang, lang_iso)

    kind = args.kind

    name = args.name

    # en_GB-alan-low.onnx.json
    config = load_config(f"{lang}-{name}-{kind}.onnx.json")

    print("generate tokens")
    generate_tokens(config)

    sample_rate = config["audio"]["sample_rate"]
    if sample_rate == 22500:
        print("Change sample rate from 22500 to 22050")
        sample_rate = 22050

    if "lang_code" in config:
        voice = config["lang_code"]
    else:
        voice = config["espeak"]["voice"]

    has_g2pw = 0
    has_espeak = 1

    if (
        "phoneme_type" in config
        and config["phoneme_type"] == "pinyin"
        and voice == "zh"
    ):
        has_espeak = 0
        has_g2pw = 1

    print("add model metadata")
    meta_data = {
        "model_type": "vits",
        "comment": "piper",  # must be piper for models from piper
        "language": lang_iso.name,
        "voice": voice,  # e.g., en-us
        "version": 1,
        "has_espeak": has_espeak,
        "has_g2pw": has_g2pw,
        "n_speakers": config["num_speakers"],
        "sample_rate": sample_rate,
    }
    print(meta_data)
    add_meta_data(f"{lang}-{name}-{kind}.onnx", meta_data)


main()
