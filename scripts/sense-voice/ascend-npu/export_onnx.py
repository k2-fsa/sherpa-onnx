#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

from typing import List, Tuple

import sentencepiece as spm
import torch

from torch_model import SenseVoiceSmall


def load_cmvn(filename) -> Tuple[List[float], List[float]]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = list(map(lambda x: float(x), t))
            else:
                inv_stddev = list(map(lambda x: float(x), t))

    return neg_mean, inv_stddev


def generate_tokens(sp):
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    print("saved to tokens.txt")


class ModelWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, prompt):
        logits = self.m(x[None], prompt)[0]
        part1 = logits[:4]
        part2 = logits[4:]
        part1 = part1.reshape(4, 25055)
        part2 = part2.reshape(x.size(0), 25055)
        return part1, part2


@torch.no_grad()
def main():
    sp = spm.SentencePieceProcessor()
    sp.load("./chn_jpn_yue_eng_ko_spectok.bpe.model")
    generate_tokens(sp)

    print("loading model")

    state_dict = torch.load("./model.pt", map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    neg_mean, inv_stddev = load_cmvn("./am.mvn")

    neg_mean = torch.tensor(neg_mean, dtype=torch.float32)
    inv_stddev = torch.tensor(inv_stddev, dtype=torch.float32)

    model = SenseVoiceSmall(neg_mean=neg_mean, inv_stddev=inv_stddev)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict

    model = ModelWrapper(model)
    model.eval()

    x = torch.randn(1, 93, 560, dtype=torch.float32)

    language = 3
    text_norm = 15
    prompt = torch.tensor([language, 1, 2, text_norm], dtype=torch.int32)

    opset_version = 14
    filename = "model.onnx"
    torch.onnx.export(
        model.m,
        (x, prompt),
        filename,
        opset_version=opset_version,
        input_names=["x", "prompt"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "logits": {0: "N", 1: "T_4"},
        },
    )
    print(f"saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(20251018)
    main()
