#!/usr/bin/env python3
# Copyright      2023  Xiaomi Corp.        (authors: Fangjun Kuang)

# pip install git+https://github.com/wenet-e2e/wenet.git
# pip install onnxruntime onnx pyyaml
# cp -a ~/open-source/wenet/wenet/transducer/search .
# cp -a ~/open-source//wenet/wenet/e_branchformer .
# cp -a ~/open-source/wenet/wenet/ctl_model .

import os
from typing import Dict

import onnx
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic

from wenet.utils.init_model import init_model


class Foo:
    pass


def add_meta_data(filename: str, meta_data: Dict[str, str]):
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


class OnnxModel(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, ctc: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc

    def forward(self, x, x_lens):
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 1-D tensor of shape (N,) containing valid lengths in x before
            padding. Its type is torch.int64
        """
        encoder_out, encoder_out_mask = self.encoder(
            x,
            x_lens,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
        )
        log_probs = self.ctc.log_softmax(encoder_out)
        log_probs_lens = encoder_out_mask.int().squeeze(1).sum(1)

        return log_probs, log_probs_lens


@torch.no_grad()
def main():
    args = Foo()
    args.checkpoint = "./final.pt"
    config_file = "./train.yaml"

    with open(config_file, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    torch_model, configs = init_model(args, configs)
    torch_model.eval()

    onnx_model = OnnxModel(encoder=torch_model.encoder, ctc=torch_model.ctc)
    filename = "model.onnx"

    N = 1
    T = 1000
    C = 80
    x = torch.rand(N, T, C, dtype=torch.float)
    x_lens = torch.full((N,), fill_value=T, dtype=torch.int64)

    opset_version = 13
    onnx_model = torch.jit.script(onnx_model)
    torch.onnx.export(
        onnx_model,
        (x, x_lens),
        filename,
        opset_version=opset_version,
        input_names=["x", "x_lens"],
        output_names=["log_probs", "log_probs_lens"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_lens": {0: "N"},
            "log_probs": {0: "N", 1: "T"},
            "log_probs_lens": {0: "N"},
        },
    )

    # https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_exp.tar.gz
    url = os.environ.get("WENET_URL", "")
    meta_data = {
        "model_type": "wenet-ctc",
        "version": "1",
        "model_author": "wenet",
        "comment": "non-streaming",
        "url": url,
    }
    add_meta_data(filename=filename, meta_data=meta_data)

    print("Generate int8 quantization models")

    filename_int8 = f"model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
