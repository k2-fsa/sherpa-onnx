#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import torch

from export_encoder_onnx import load_model, get_num_input_frames

import argparse


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-len-in-seconds",
        type=int,
        required=True,
        help="""RKNN/QNN does not support dynamic shape, so we need to hard-code
        how long the model can process.
        """,
    )

    parser.add_argument(
        "--float-mask",
        type=int,
        default=1,
        help="1 to use float mask. 0 to use int32 mask",
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    args = get_args()

    input_len_in_seconds = int(args.input_len_in_seconds)
    num_input_frames = get_num_input_frames(input_len_in_seconds)

    encoder_out = torch.randn(1, num_input_frames, 512, dtype=torch.float32)
    acoustic_embedding = torch.randn(1, num_input_frames, 512, dtype=torch.float32)
    if args.float_mask == 1:
        mask = torch.ones([num_input_frames], dtype=torch.float32)
    else:
        mask = torch.ones([num_input_frames], dtype=torch.int32)

    d = model.decoder(encoder_out, acoustic_embedding)
    print("d", d.shape)

    opset_version = args.opset_version
    filename = f"decoder-{input_len_in_seconds}-seconds.onnx"
    torch.onnx.export(
        model.decoder,
        (encoder_out, acoustic_embedding, mask),
        filename,
        opset_version=opset_version,
        input_names=["encoder_out", "acoustic_embedding", "mask"],
        output_names=["decoder_out"],
        dynamic_axes={},
    )
    print(f"Saved to {filename}")


if __name__ == "__main__":
    torch.manual_seed(20251008)
    main()
