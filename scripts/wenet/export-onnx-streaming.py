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


def add_meta_data(filename: str, meta_data: Dict[str, str]):
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

    #  model = onnx.version_converter.convert_version(model, 21)

    onnx.save(model, filename)


class OnnxModel(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, ctc: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc

    def forward(
        self,
        x: torch.Tensor,
        offset: torch.Tensor,
        required_cache_size: torch.Tensor,
        attn_cache: torch.Tensor,
        conv_cache: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        """
        Args:
          x:
            A 3-D float32 tensor of shape (N, T, C). It supports only N == 1.
          offset:
            A scalar of dtype torch.int64.
          required_cache_size:
            A scalar of dtype torch.int64.
          attn_cache:
            A 4-D float32 tensor of shape (num_blocks, head, required_cache_size, encoder_output_size / head /2).
          conv_cache:
            A 4-D float32 tensor of shape (num_blocks, N, encoder_output_size, cnn_module_kernel - 1).
          attn_mask:
            A 3-D bool tensor of shape (N, 1, required_cache_size + chunk_size)
        Returns:
          Return a tuple of 3 tensors:
            - A 3-D float32 tensor of shape (N, T, C) containing log_probs
            - next_attn_cache
            - next_conv_cache
        """
        encoder_out, next_att_cache, next_conv_cache = self.encoder.forward_chunk(
            xs=x,
            offset=offset,
            required_cache_size=required_cache_size,
            att_cache=attn_cache,
            cnn_cache=conv_cache,
            att_mask=attn_mask,
        )
        log_probs = self.ctc.log_softmax(encoder_out)

        return log_probs, next_att_cache, next_conv_cache


class Foo:
    pass


@torch.no_grad()
def main():
    args = Foo()
    args.checkpoint = "./final.pt"
    config_file = "./train.yaml"

    with open(config_file, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    torch_model, configs = init_model(args, configs)
    torch_model.eval()

    head = configs["encoder_conf"]["attention_heads"]
    num_blocks = configs["encoder_conf"]["num_blocks"]
    output_size = configs["encoder_conf"]["output_size"]
    cnn_module_kernel = configs["encoder_conf"].get("cnn_module_kernel", 1)

    right_context = torch_model.right_context()
    subsampling_factor = torch_model.encoder.embed.subsampling_rate
    chunk_size = 16
    left_chunks = 4

    decoding_window = (chunk_size - 1) * subsampling_factor + right_context + 1

    required_cache_size = chunk_size * left_chunks

    offset = required_cache_size

    attn_cache = torch.zeros(
        num_blocks,
        head,
        required_cache_size,
        output_size // head * 2,
        dtype=torch.float32,
    )

    attn_mask = torch.ones(1, 1, required_cache_size + chunk_size, dtype=torch.bool)
    attn_mask[:, :, :required_cache_size] = 0

    conv_cache = torch.zeros(
        num_blocks, 1, output_size, cnn_module_kernel - 1, dtype=torch.float32
    )

    sos = torch_model.sos_symbol()
    eos = torch_model.eos_symbol()

    onnx_model = OnnxModel(
        encoder=torch_model.encoder,
        ctc=torch_model.ctc,
    )
    filename = "model-streaming.onnx"

    N = 1
    T = decoding_window
    C = 80
    x = torch.rand(N, T, C, dtype=torch.float32)
    offset = torch.tensor([offset], dtype=torch.int64)
    required_cache_size = torch.tensor([required_cache_size], dtype=torch.int64)

    opset_version = 13
    torch.onnx.export(
        onnx_model,
        (x, offset, required_cache_size, attn_cache, conv_cache, attn_mask),
        filename,
        opset_version=opset_version,
        input_names=[
            "x",
            "offset",
            "required_cache_size",
            "attn_cache",
            "conv_cache",
            "attn_mask",
        ],
        output_names=["log_probs", "next_att_cache", "next_conv_cache"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "attn_cache": {2: "T"},
            "attn_mask": {2: "T"},
            "log_probs": {0: "N"},
            "new_attn_cache": {2: "T"},
        },
    )

    # https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_exp.tar.gz
    url = os.environ.get("WENET_URL", "")
    meta_data = {
        "model_type": "wenet_ctc",
        "version": "1",
        "model_author": "wenet",
        "comment": "streaming",
        "url": "https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_exp.tar.gz",
        "chunk_size": chunk_size,
        "left_chunks": left_chunks,
        "head": head,
        "num_blocks": num_blocks,
        "output_size": output_size,
        "cnn_module_kernel": cnn_module_kernel,
        "right_context": right_context,
        "subsampling_factor": subsampling_factor,
        "vocab_size": torch_model.ctc.ctc_lo.weight.shape[0],
    }
    add_meta_data(filename=filename, meta_data=meta_data)

    print("Generate int8 quantization models")

    filename_int8 = f"model-streaming.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
    )


if __name__ == "__main__":
    main()
