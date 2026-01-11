#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import glob
from pathlib import Path

import numpy as np
import torch

from export_encoder_onnx import get_args, get_num_input_frames, load_model
from export_predictor_onnx import modified_predictor_forward
from test_onnx import compute_feat, get_acoustic_embedding
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

        encoder_out = model.encoder(f)
        alpha = model.predictor(encoder_out)

        acoustic_embedding = get_acoustic_embedding(
            alpha[0].numpy(), encoder_out[0].numpy()
        )
        acoustic_embedding = torch.from_numpy(acoustic_embedding[None])
        num_tokens = acoustic_embedding.shape[1]

        acoustic_embedding = torch.nn.functional.pad(
            acoustic_embedding,
            (0, 0, 0, encoder_out.shape[1] - num_tokens),
            "constant",
            0,
        )

        mask = torch.zeros(1, encoder_out.shape[1], dtype=torch.int32)

        mask[0, :num_tokens] = 1

        # NOTE(Fangjun): We have to transpose the data since QNN expects
        # (N, C, T) for the decoder model
        # Not sure why it has such a requirement.

        encoder_out = encoder_out.permute(0, 2, 1).clone().numpy()
        acoustic_embedding = acoustic_embedding.permute(0, 2, 1).clone().numpy()

        print("inputs: ", encoder_out.shape, acoustic_embedding.shape, mask.shape)

        name = Path(w).stem

        first = f"decoder-input-{name}-0.raw"
        second = f"decoder-input-{name}-1.raw"
        third = f"decoder-input-{name}-2.raw"
        encoder_out.tofile(first)
        acoustic_embedding.tofile(second)
        mask.numpy().tofile(third)

        name_list.append((first, second, third))

    with open("decoder-input-list.txt", "w") as f:
        for first, second, third in name_list:
            f.write(f"{first} {second} {third}\n")


if __name__ == "__main__":
    main()
