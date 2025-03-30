#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import argparse

import onnx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-model", type=str, required=True, help="input onnx model")

    parser.add_argument(
        "--out-model", type=str, required=True, help="output onnx model"
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args.in_model, args.out_model)

    model = onnx.load(args.in_model)

    meta_data = {
        "model_type": "vocos",
        "model_filename": "mel_spec_22khz_univ.onnx",
        "sample_rate": 22050,
        "version": 1,
        "model_author": "BSC-LT",
        "maintainer": "k2-fsa",
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "window_type": "hann",
        "center": 1,
        "pad_mode": "reflect",
        "normalized": 0,
        "url1": "https://huggingface.co/BSC-LT/vocos-mel-22khz",
        "url2": "https://github.com/gemelo-ai/vocos",
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

    onnx.save(model, args.out_model)

    print(f"Saved to {args.out_model}")


if __name__ == "__main__":
    main()
