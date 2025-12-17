#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import torch
import base64

from nano import Nano
from test_onnx import compute_feat, load_audio


def load_tokens(filename):
    id2token = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            try:
                f = line.strip().split()
                if len(f) == 2:
                    t, i = f
                else:
                    t = " "
                    i = f[0]
                id2token[int(i)] = t
            except Exception as ex:
                print(ex)
                raise
    return id2token


def load_torch_model():
    if not Path("./model.pt").is_file():
        raise ValueError(
            "Please download files from https://huggingface.co/csukuangfj/funasr-nano-with-ctc"
        )
    model = Nano()

    state_dict = torch.load("./model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    del state_dict

    return model


@torch.no_grad()
def main():
    model = load_torch_model()

    samples, sample_rate = load_audio("./zh.wav")
    assert sample_rate == 16000, sample_rate

    features = compute_feat(samples=samples, sample_rate=sample_rate)
    x = torch.from_numpy(features)[None]
    logits = model(x)

    idx = logits.squeeze(0).argmax(dim=-1)
    print(idx)
    idx = torch.unique_consecutive(idx).tolist()
    print(idx)

    id2token = load_tokens("./tokens.txt")
    blank_id = len(id2token) - 1

    idx = [i for i in idx if i != blank_id]
    print(idx)

    text = "".join([id2token[i] for i in idx])

    s = b""
    for i in idx:
        s += base64.b64decode(id2token[i])

    text = s.decode().strip()
    print(text)


if __name__ == "__main__":
    main()
