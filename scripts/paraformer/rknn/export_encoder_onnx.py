#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import argparse
import os
from typing import Any, Dict, List, Tuple

import onnx
import torch
import yaml

from torch_model import Paraformer, SANMEncoder


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-len-in-seconds",
        type=int,
        required=True,
        help="""RKNN does not support dynamic shape, so we need to hard-code
        how long the model can process.
        """,
    )
    return parser.parse_args()


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


if __name__ == "__main__":

    def modified_sanm_encoder_forward(
        self: SANMEncoder, xs_pad: torch.Tensor, pos: torch.Tensor
    ):
        print("xs pad", xs_pad.shape)
        xs_pad = (xs_pad + self.neg_mean) * self.inv_stddev

        xs_pad = xs_pad * self.output_size() ** 0.5

        xs_pad = xs_pad + pos

        xs_pad = self.encoders0(xs_pad)[0]

        xs_pad = self.encoders(xs_pad)[0]

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        print("xs pad--->", xs_pad.shape, pos.shape)

        return xs_pad

    #  SANMEncoder.forward = modified_sanm_encoder_forward


def load_model():
    with open("./config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("creating model")

    neg_mean, inv_stddev = load_cmvn("./am.mvn")

    neg_mean = torch.tensor(neg_mean, dtype=torch.float32)
    inv_stddev = torch.tensor(inv_stddev, dtype=torch.float32)

    m = Paraformer(
        neg_mean=neg_mean,
        inv_stddev=inv_stddev,
        input_size=560,
        vocab_size=8404,
        encoder_conf=config["encoder_conf"],
        decoder_conf=config["decoder_conf"],
        predictor_conf=config["predictor_conf"],
    )
    m.eval()

    print("loading state dict")
    state_dict = torch.load("./model_state_dict.pt", map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    m.load_state_dict(state_dict)
    del state_dict

    return m


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


lfr_window_size = 7
lfr_window_shift = 6


def get_num_input_frames(input_len_in_seconds):
    num_frames = input_len_in_seconds * 100
    print("num_frames", num_frames)

    # num_input_frames is an approximate number
    num_input_frames = int(num_frames / lfr_window_shift + 0.5)
    print("num_input_frames", num_input_frames)
    return num_input_frames


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    print("loading model")
    model = load_model()

    # frame shift is 10ms, 1 second has about 100 feature frames
    input_len_in_seconds = int(args.input_len_in_seconds)
    num_input_frames = get_num_input_frames(input_len_in_seconds)

    x = torch.randn(1, num_input_frames, 560, dtype=torch.float32)
    pos_emb = torch.rand(1, x.shape[1], 560, dtype=torch.float32)

    opset_version = 14
    filename = f"encoder-{input_len_in_seconds}-seconds.onnx"
    torch.onnx.export(
        model.encoder,
        #  (x, pos_emb),
        x,
        filename,
        opset_version=opset_version,
        #  input_names=["x", "pos_emb"],
        input_names=["x"],
        output_names=["encoder_out"],
        dynamic_axes={},
    )

    model_author = os.environ.get("model_author", "iic")
    comment = os.environ.get(
        "comment",
        "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    url = os.environ.get("url", "https://github.com/alibaba-damo-academy/FunASR")

    meta_data = {
        "lfr_window_size": lfr_window_size,
        "lfr_window_shift": lfr_window_shift,
        "num_input_frames": num_input_frames,
        "normalize_samples": 0,  # input should be in the range [-32768, 32767]
        "model_type": "paraformer",
        "version": "1",
        "model_author": model_author,
        "maintainer": "k2-fsa",
        "vocab_size": 8404,
        "comment": comment,
        "url": url,
        "rknn": 1,
    }

    add_meta_data(filename=filename, meta_data=meta_data)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(20251013)
    main()
