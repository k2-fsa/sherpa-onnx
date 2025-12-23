#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import numpy as np
import torch

from export_encoder_onnx import load_model
from export_predictor_onnx import modified_predictor_forward
from test_onnx import get_acoustic_embedding
from torch_model import CifPredictorV2

CifPredictorV2.forward = modified_predictor_forward


def load_tokens():
    id2token = dict()
    with open("./tokens.txt") as f:
        for line in f:
            fields = line.strip().split()
            id2token[int(fields[1])] = fields[0]
    return id2token


@torch.no_grad()
def main():
    model = load_model()
    features = np.load
    features = np.fromfile("./encoder-input-zh.raw", dtype=np.float32).reshape(
        (1, -1, 560)
    )
    features = torch.from_numpy(features)
    encoder_out = model.encoder(features)
    encoder_out.permute(0, 2, 1).numpy().tofile("predictor-in.raw")

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

    mask = torch.zeros(1, encoder_out.shape[1], dtype=torch.float32)

    mask[0, :num_tokens] = 1
    logits = model.decoder(encoder_out, acoustic_embedding, mask)
    yseq = logits[0, :num_tokens].argmax(axis=-1).tolist()
    print(yseq, "-->", len(yseq))

    id2token = load_tokens()
    text = [id2token[i] for i in yseq]
    print(text)

    qnn_encoder_out = np.fromfile("./encoder_out.raw", dtype=np.float32).reshape(
        1, -1, 512
    )

    qnn_encoder_out = torch.from_numpy(qnn_encoder_out)

    qnn_alpha = np.fromfile("./alphas.raw", dtype=np.float32).reshape(1, -1)
    qnn_alpha = torch.from_numpy(qnn_alpha)

    acoustic_embedding = get_acoustic_embedding(
        qnn_alpha[0].numpy(), qnn_encoder_out[0].numpy()
    )
    acoustic_embedding = torch.from_numpy(acoustic_embedding[None])

    num_tokens = acoustic_embedding.shape[1]

    acoustic_embedding = torch.nn.functional.pad(
        acoustic_embedding,
        (0, 0, 0, qnn_encoder_out.shape[1] - num_tokens),
        "constant",
        0,
    )

    mask = torch.zeros(1, qnn_encoder_out.shape[1], dtype=torch.float32)

    mask[0, :num_tokens] = 1
    logits = model.decoder(qnn_encoder_out, acoustic_embedding, mask)
    yseq = logits[0, :num_tokens].argmax(axis=-1).tolist()
    print(yseq, "-->", len(yseq))
    text = [id2token[i] for i in yseq]
    print(text)


if __name__ == "__main__":
    torch.manual_seed(20251013)
    main()
