#!/usr/bin/env python3
# Copyright (c)  2025  Xiaomi Corporation

import torch

from export_encoder_onnx import load_model, get_args, get_num_input_frames


@torch.no_grad()
def main():
    print("loading model")
    model = load_model()

    args = get_args()

    input_len_in_seconds = int(args.input_len_in_seconds)
    num_input_frames = get_num_input_frames(input_len_in_seconds)

    encoder_out = torch.randn(1, num_input_frames, 512, dtype=torch.float32)
    acoustic_embedding = torch.randn(1, num_input_frames, 512, dtype=torch.float32)
    mask = torch.ones([num_input_frames], dtype=torch.float32)

    d = model.decoder(encoder_out, acoustic_embedding)
    print("d", d.shape)

    opset_version = 14
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
