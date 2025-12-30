#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import glob
from pathlib import Path

import numpy as np

from export_encoder_onnx import get_args, get_num_input_frames
from test_onnx import compute_feat


def pad(features, max_len):
    if features.shape[0] > max_len:
        return features[:max_len]
    elif features.shape[0] < max_len:
        features = np.pad(
            features,
            ((0, max_len - features.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return features


def main():
    args = get_args()
    print(vars(args))

    input_len_in_seconds = int(args.input_len_in_seconds)
    num_input_frames = get_num_input_frames(input_len_in_seconds)

    wav_files = glob.glob("*.wav")
    features_name = []
    for w in wav_files:
        f = compute_feat(w)
        print(w, f.shape)
        f = pad(f, num_input_frames)
        print(f.shape)
        print()
        name = Path(w).stem

        s = f"encoder-input-{name}.raw"
        f.tofile(s)
        features_name.append(s)

    with open("encoder-input-list.txt", "w") as f:
        for line in features_name:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
