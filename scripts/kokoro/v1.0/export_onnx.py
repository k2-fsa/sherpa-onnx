#!/usr/bin/env python3

import json

import torch
from kokoro import KModel
from kokoro.model import KModelForONNX


@torch.no_grad()
def main():
    with open("Kokoro-82M/config.json") as f:
        config = json.load(f)

    model = (
        KModel(
            repo_id="not-used-any-value-is-ok",
            model="Kokoro-82M/kokoro-v1_0.pth",
            config=config,
            disable_complex=True,
        )
        .to("cpu")
        .eval()
    )

    x = torch.randint(1, 100, (48,)).numpy()
    x = torch.LongTensor([[0, *x, 0]])

    style = torch.rand(1, 256, dtype=torch.float32)
    speed = torch.rand(1)

    print(x.shape, x.dtype)
    print(style.shape, style.dtype)
    print(speed, speed.dtype)

    model2 = KModelForONNX(model)

    torch.onnx.export(
        model2,
        (x, style, speed),
        "kokoro.onnx",
        input_names=["tokens", "style", "speed"],
        output_names=["audio"],
        dynamic_axes={
            "tokens": {1: "sequence_length"},
            "audio": {0: "audio_length"},
        },
        opset_version=14,  # minimum working version for this kokoro model is 14
    )


if __name__ == "__main__":
    main()
