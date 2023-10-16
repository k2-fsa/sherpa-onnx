#!/usr/bin/env python3
# Copyright    2023  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
This script converts vits models trained using the VCTK dataset.

Usage:

(1) Download vits

cd /Users/fangjun/open-source
git clone https://github.com/jaywalnut310/vits

(2) Download pre-trained models from
https://huggingface.co/csukuangfj/vits-vctk/tree/main

wget https://huggingface.co/csukuangfj/vits-vctk/resolve/main/pretrained_vctk.pth

(3) Run this file

./export-onnx-vctk.py  \
  --config ~/open-source//vits/configs/vctk_base.json \
  --checkpoint ~/open-source/icefall-models/vits-vctk/pretrained_vctk.pth

It will generate the following two files:

$ ls -lh *.onnx
-rw-r--r--  1 fangjun  staff    37M Oct 16 10:57 vits-vctk.int8.onnx
-rw-r--r--  1 fangjun  staff   116M Oct 16 10:57 vits-vctk.onnx
"""
import sys

# Please change this line to point to the vits directory.
# You can download vits from
# https://github.com/jaywalnut310/vits
sys.path.insert(0, "/Users/fangjun/open-source/vits")  # noqa

import argparse
from pathlib import Path
from typing import Dict, Any

import commons
import onnx
import torch
import utils
from models import SynthesizerTrn
from onnxruntime.quantization import QuantType, quantize_dynamic
from text import text_to_sequence
from text.symbols import symbols
from text.symbols import _punctuation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="""Path to vctk_base.json.
        You can find it at
        https://huggingface.co/csukuangfj/vits-vctk/resolve/main/vctk_base.json
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="""Path to the checkpoint file.
        You can find it at
        https://huggingface.co/csukuangfj/vits-vctk/resolve/main/pretrained_vctk.pth
        """,
    )

    return parser.parse_args()


class OnnxModel(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        x_lengths,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        sid=0,
        max_len=None,
    ):
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            max_len=max_len,
        )[0]


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def check_args(args):
    assert Path(args.config).is_file(), args.config
    assert Path(args.checkpoint).is_file(), args.checkpoint


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
    with open("tokens-vctk.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(symbols):
            f.write(f"{s} {i}\n")
    print("Generated tokens-vctk.txt")


@torch.no_grad()
def main():
    args = get_args()
    check_args(args)

    generate_tokens()

    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.checkpoint, net_g, None)

    x = get_text("Liliana is the most beautiful assistant", hps)
    x = x.unsqueeze(0)

    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    length_scale = torch.tensor([1], dtype=torch.float32)
    noise_scale_w = torch.tensor([1], dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.int64)

    model = OnnxModel(net_g)

    opset_version = 13

    filename = "vits-vctk.onnx"

    torch.onnx.export(
        model,
        (x, x_length, noise_scale, length_scale, noise_scale_w, sid),
        filename,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_length",
            "noise_scale",
            "length_scale",
            "noise_scale_w",
            "sid",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},  # n_audio is also known as batch_size
            "x_length": {0: "N"},
            "y": {0: "N", 2: "L"},
        },
    )
    meta_data = {
        "model_type": "vits",
        "comment": "vctk",
        "language": "English",
        "add_blank": int(hps.data.add_blank),
        "n_speakers": int(hps.data.n_speakers),
        "sample_rate": hps.data.sampling_rate,
        "punctuation": " ".join(list(_punctuation)),
    }
    print("meta_data", meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)

    print("Generate int8 quantization models")

    filename_int8 = "vits-vctk.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QUInt8,
    )

    print(f"Saved to {filename} and {filename_int8}")


if __name__ == "__main__":
    main()
