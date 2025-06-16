#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json
from typing import Any, Dict

import onnx
from iso639 import Lang


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
            f.write(f"{s} {i[0]}\n")
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

    print("add model metadata")
    meta_data = {
        "model_type": "vits",
        "comment": "piper",  # must be piper for models from piper
        "language": lang_iso.name,
        "voice": config["espeak"]["voice"],  # e.g., en-us
        "has_espeak": 1,
        "n_speakers": config["num_speakers"],
        "sample_rate": sample_rate,
    }
    print(meta_data)
    add_meta_data(f"{lang}-{name}-{kind}.onnx", meta_data)


main()
