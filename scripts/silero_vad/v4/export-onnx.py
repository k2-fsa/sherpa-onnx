#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import onnx
import torch
from onnxsim import simplify

import torch
from torch import Tensor


def simple_pad(x: Tensor, pad: int) -> Tensor:
    #  _0 = torch.slice(torch.slice(torch.slice(x), 1), 2, 1, torch.add(1, pad))
    _0 = x[:, :, 1 : 1 + pad]

    left_pad = torch.flip(_0, [-1])
    #  _1 = torch.slice(torch.slice(torch.slice(x), 1), 2, torch.sub(-1, pad), -1)

    _1 = x[:, :, (-1 - pad) : -1]

    right_pad = torch.flip(_1, [-1])
    _2 = torch.cat([left_pad, x, right_pad], 2)
    return _2


class MyModule(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def adaptive_normalization_forward(self, spect):
        m = self.m._model.adaptive_normalization
        _0 = simple_pad

        # Note(fangjun): rknn uses fp16 by default, whose max value is 65504
        # so we need to re-write the computation for spect0
        #  spect0 = torch.log1p(torch.mul(spect, 1048576))
        spect0 = torch.log1p(spect) + 13.86294

        _1 = torch.eq(len(spect0.shape), 2)
        if _1:
            _2 = torch.unsqueeze(spect0, 0)
            spect1 = _2
        else:
            spect1 = spect0
        mean = torch.mean(spect1, [1], True)
        to_pad = m.to_pad
        mean0 = _0(
            mean,
            to_pad,
        )
        filter_ = m.filter_
        mean1 = torch.conv1d(mean0, filter_)
        mean_mean = torch.mean(mean1, [-1], True)
        spect2 = torch.add(spect1, torch.neg(mean_mean))
        return spect2

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        m = self.m._model

        feature_extractor = m.feature_extractor
        x0 = (feature_extractor).forward(
            x,
        )
        norm = self.adaptive_normalization_forward(x0)
        x1 = torch.cat([x0, norm], 1)
        first_layer = m.first_layer
        x2 = (first_layer).forward(
            x1,
        )
        encoder = m.encoder
        x3 = (encoder).forward(
            x2,
        )
        decoder = m.decoder
        x4, h0, c0, = (decoder).forward(
            x3,
            h,
            c,
        )
        _0 = torch.mean(torch.squeeze(x4, 1), [1])
        out = torch.unsqueeze(_0, 1)
        return (out, h0, c0)


@torch.no_grad()
def main():
    m = torch.jit.load("./silero_vad.jit")
    m = MyModule(m)
    x = torch.rand((1, 512), dtype=torch.float32)
    h = torch.rand((2, 1, 64), dtype=torch.float32)
    c = torch.rand((2, 1, 64), dtype=torch.float32)
    m = torch.jit.script(m)
    torch.onnx.export(
        m,
        (x, h, c),
        "m.onnx",
        input_names=["x", "h", "c"],
        output_names=["prob", "next_h", "next_c"],
    )

    print("simplifying ...")
    model = onnx.load("m.onnx")

    meta_data = {
        "model_type": "silero-vad-v4",
        "sample_rate": 16000,
        "version": 4,
        "h_shape": "2,1,64",
        "c_shape": "2,1,64",
    }

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    print("--------------------")
    print(model.metadata_props)

    model_simp, check = simplify(model)
    onnx.save(model_simp, "m.onnx")


if __name__ == "__main__":
    main()
