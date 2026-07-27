#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)

import inspect
import sys
from pathlib import Path
from typing import Any, Dict

import commons
import onnx
import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from torch import nn


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        assert model.use_sdp is False

    def forward(self, x, x_length, noise_scale=0.667, length_scale=1.0):
        """
        Args:
          x: (N, num_tokens), torch.int64
          x_length: (N,) torch.int64
          noise_scale: (1,), torch.float32
          length_scale: (1,), torch.float32
        """

        hidden, m_p, logs_p, x_mask = self.model.enc_p(x, x_length)
        logw = self.model.dp(hidden, x_mask, g=None)
        durations = torch.ceil(torch.exp(logw) * x_mask * length_scale)
        y_lengths = torch.clamp_min(torch.sum(durations, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attention_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attention = commons.generate_path(durations, attention_mask)
        m_p_exp = torch.matmul(
            attention.squeeze(1),
            m_p.transpose(1, 2),
        ).transpose(1, 2)
        logs_p_exp = torch.matmul(
            attention.squeeze(1),
            logs_p.transpose(1, 2),
        ).transpose(1, 2)

        z_p = m_p_exp + torch.randn_like(m_p_exp) * torch.exp(logs_p_exp) * noise_scale

        z = self.model.flow(z_p, y_mask, g=None, reverse=True)

        return self.model.dec(z * y_mask, g=None)


def load_model(checkpoint_dir: Path) -> nn.Module:
    hps = utils.get_hparams_from_file(str(checkpoint_dir / "config.json"))
    model = (
        SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        )
        .cpu()
        .eval()
    )
    utils.load_checkpoint(str(checkpoint_dir / "model.pth"), model, None)
    return model


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def generate_tokens():
    # WARNING(fangjun): There are duplicate tokens in symbols
    token2id = {sym: idx for idx, sym in enumerate(symbols)}
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for s, i in token2id.items():
            f.write(f"{s} {i}\n")


@torch.no_grad()
def export(checkpoint_dir: Path) -> None:
    model = load_model(checkpoint_dir)
    model = model.eval()

    model_path = "./model.onnx"

    x = torch.tensor(
        [[0, 18, 0, 61, 0, 55, 0, 48, 0, 44, 0, 46, 0]],
        dtype=torch.int64,
    )
    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([0.667], dtype=torch.float32)
    length_scale = torch.tensor([1.0], dtype=torch.float32)

    export_sig = inspect.signature(torch.onnx.export)

    kwargs = dict()
    if "dynamo" in export_sig.parameters:
        kwargs["dynamo"] = False

    if "external_data" in export_sig.parameters:
        kwargs["external_data"] = False

    torch.onnx.export(
        ModelWrapper(model),
        (x, x_length, noise_scale, length_scale),
        model_path,
        input_names=["x", "x_length", "noise_scale", "length_scale"],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_length": {0: "N"},
            "y": {0: "N", 1: "num_samples"},
        },
        opset_version=13,
        do_constant_folding=True,
        **kwargs,
    )

    _punctuation = ';:,.!?¡¿—…"«»“” '
    meta_data = {
        "model_type": "vits",
        "comment": "Inflect",
        "language": "English",
        "add_blank": 1,
        "n_speakers": 1,
        "sample_rate": 24000,
        "punctuation": " ".join(list(_punctuation)),
        "author": "https://huggingface.co/owensong",
        "URL": f"https://huggingface.co/owensong/{checkpoint_dir}",
        "license": "Apache-2.0",
        "voice": "en-US",
        "has_espeak": "1",
        "version": 2,
    }
    print(meta_data)
    add_meta_data(filename=str(model_path), meta_data=meta_data)


def main() -> None:
    torch.manual_seed(42)

    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: python3 ./export_onnx.py ./path/to/checkpoint_dir")

    generate_tokens()

    checkpoint_dir = Path(sys.argv[1])
    export(checkpoint_dir)


if __name__ == "__main__":
    main()
