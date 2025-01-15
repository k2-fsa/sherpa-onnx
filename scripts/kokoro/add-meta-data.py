#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import argparse
import json
from pathlib import Path

import numpy as np
import onnx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="input and output onnx model"
    )

    parser.add_argument("--voices", type=str, required=True, help="Path to voices.json")
    return parser.parse_args()


def load_voices(filename):
    with open(filename) as f:
        voices = json.load(f)
    for key in voices:
        voices[key] = np.array(voices[key], dtype=np.float32)
    return voices


def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts


def generate_tokens():
    token2id = get_vocab()
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for s, i in token2id.items():
            f.write(f"{s} {i}\n")


def main():
    args = get_args()
    print(args.model, args.voices)

    model = onnx.load(args.model)
    voices = load_voices(args.voices)

    if Path("./tokens.txt").is_file():
        print("./tokens.txt exist, skip generating it")
    else:
        generate_tokens()

    keys = list(voices.keys())
    print(",".join(keys))

    if Path("./voices.bin").is_file():
        print("./voices.bin exists, skip generating it")
    else:
        with open("voices.bin", "wb") as f:
            for k in keys:
                f.write(voices[k].tobytes())

    meta_data = {
        "model_type": "kokoro",
        "language": "English",
        "has_espeak": 1,
        "sample_rate": 24000,
        "version": 1,
        "voice": "en-us",
        "style_dim": ",".join(map(str, voices[keys[0]].shape)),
        "n_speakers": len(keys),
        "speaker_names": ",".join(keys),
        "model_url": "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files",
        "see_also": "https://huggingface.co/spaces/hexgrad/Kokoro-TTS",
        "see_also_2": "https://huggingface.co/hexgrad/Kokoro-82M",
        "maintainer": "k2-fsa",
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
