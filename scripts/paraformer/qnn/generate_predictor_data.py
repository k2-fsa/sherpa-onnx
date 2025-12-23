#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import glob
from pathlib import Path

import numpy as np
import torch

from export_encoder_onnx import get_args, get_num_input_frames, load_model
from export_predictor_onnx import modified_predictor_forward
from test_onnx import compute_feat
from torch_model import CifPredictorV2

CifPredictorV2.forward = modified_predictor_forward


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


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    input_len_in_seconds = int(args.input_len_in_seconds)
    num_input_frames = get_num_input_frames(input_len_in_seconds)

    wav_files = glob.glob("*.wav")

    model = load_model()

    name_list = []
    for w in wav_files:
        f = compute_feat(w)
        print(w, f.shape)
        f = pad(f, num_input_frames)
        f = f[None]
        print(f.shape)

        f = torch.from_numpy(f)

        encoder_out = model.encoder(f).numpy()

        name = Path(w).stem

        s = f"encoder-output-{name}.raw"
        encoder_out.tofile(s)
        name_list.append(s)
        print(encoder_out.shape)

    with open("encoder-output-list.txt", "w") as f:
        for line in name_list:
            f.write(f"{line}\n")


if __name__ == "__main__":
    main()
