#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
<|en|>
<|pnc|>
<|noitn|>
<|nodiarize|>
<|notimestamp|>
"""

import os
from typing import Dict, Tuple

import nemo
import onnx
import torch
from nemo.collections.common.parts import NEG_INF
from onnxruntime.quantization import QuantType, quantize_dynamic

"""
NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED :
Could not find an implementation for Trilu(14) node with name '/Trilu'

See also https://github.com/microsoft/onnxruntime/issues/16189#issuecomment-1722219631

So we use fixed_form_attention_mask() to replace
the original form_attention_mask()
"""


def fixed_form_attention_mask(input_mask, diagonal=None):
    """
    Fixed: Build attention mask with optional masking of future tokens we forbid
    to attend to (e.g. as it is in Transformer decoder).

    Args:
        input_mask: binary mask of size B x L with 1s corresponding to valid
            tokens and 0s corresponding to padding tokens
        diagonal: diagonal where triangular future mask starts
            None -- do not mask anything
            0 -- regular translation or language modeling future masking
            1 -- query stream masking as in XLNet architecture
    Returns:
        attention_mask: mask of size B x 1 x L x L with 0s corresponding to
            tokens we plan to attend to and -10000 otherwise
    """

    if input_mask is None:
        return None
    attn_shape = (1, input_mask.shape[1], input_mask.shape[1])
    attn_mask = input_mask.to(dtype=bool).unsqueeze(1)
    if diagonal is not None:
        future_mask = torch.tril(
            torch.ones(
                attn_shape,
                dtype=torch.int64,  # it was torch.bool
                # but onnxruntime does not support torch.int32 or torch.bool
                # in torch.tril
                device=input_mask.device,
            ),
            diagonal,
        ).bool()
        attn_mask = attn_mask & future_mask
    attention_mask = (1 - attn_mask.to(torch.float)) * NEG_INF
    return attention_mask.unsqueeze(1)


nemo.collections.common.parts.form_attention_mask = fixed_form_attention_mask

from nemo.collections.asr.models import EncDecMultiTaskModel


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename, load_external_data=False)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def lens_to_mask(lens, max_length):
    """
    Create a mask from a tensor of lengths.
    """
    batch_size = lens.shape[0]
    arange = torch.arange(max_length, device=lens.device)
    mask = arange.expand(batch_size, max_length) < lens.unsqueeze(1)
    return mask


class EncoderWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.encoder
        self.encoder_decoder_proj = m.encoder_decoder_proj

    def forward(
        self, x: torch.Tensor, x_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (N, T, C)
          x_len: (N,)
        Returns:
          - enc_states: (N, T, C)
          - encoded_len: (N,)
          - enc_mask: (N, T)
        """
        x = x.permute(0, 2, 1)
        # x: (N, C, T)
        encoded, encoded_len = self.encoder(audio_signal=x, length=x_len)

        enc_states = encoded.permute(0, 2, 1)

        enc_states = self.encoder_decoder_proj(enc_states)

        enc_mask = lens_to_mask(encoded_len, enc_states.shape[1])

        return enc_states, encoded_len, enc_mask


class DecoderWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.decoder = m.transf_decoder
        self.log_softmax = m.log_softmax

        # We use only greedy search, so there is no need to compute log_softmax
        self.log_softmax.mlp.log_softmax = False

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_mems_list_0: torch.Tensor,
        decoder_mems_list_1: torch.Tensor,
        decoder_mems_list_2: torch.Tensor,
        decoder_mems_list_3: torch.Tensor,
        decoder_mems_list_4: torch.Tensor,
        decoder_mems_list_5: torch.Tensor,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor,
    ):
        """
        Args:
          decoder_input_ids: (N, num_tokens), torch.int32
          decoder_mems_list_i: (N, num_tokens, 1024)
          enc_states: (N, T, 1024)
          enc_mask: (N, T)
        Returns:
          - logits: (N, 1, vocab_size)
          - decoder_mems_list_i: (N, num_tokens_2, 1024)
        """
        pos = decoder_input_ids[0][-1].item()
        decoder_input_ids = decoder_input_ids[:, :-1]

        decoder_hidden_states = self.decoder.embedding.forward(
            decoder_input_ids, start_pos=pos
        )
        decoder_input_mask = torch.ones_like(decoder_input_ids).float()

        decoder_mems_list, _xatt_scores = self.decoder.decoder.forward(
            decoder_hidden_states,
            decoder_input_mask,
            enc_states,
            enc_mask,
            [
                decoder_mems_list_0,
                decoder_mems_list_1,
                decoder_mems_list_2,
                decoder_mems_list_3,
                decoder_mems_list_4,
                decoder_mems_list_5,
            ],
            return_mems=True,
        )
        (
            out_mems_0,
            out_mems_1,
            out_mems_2,
            out_mems_3,
            out_mems_4,
            out_mems_5,
        ) = decoder_mems_list
        logits = self.log_softmax(hidden_states=out_mems_5[:, -1:])

        return (
            logits,
            out_mems_0,
            out_mems_1,
            out_mems_2,
            out_mems_3,
            out_mems_4,
            out_mems_5,
        )


def export_encoder(canary_model):
    encoder = EncoderWrapper(canary_model)
    x = torch.rand(1, 4000, 128)
    x_lens = torch.tensor([x.shape[1]], dtype=torch.int64)

    encoder_filename = "encoder.onnx"
    torch.onnx.export(
        encoder,
        (x, x_lens),
        encoder_filename,
        input_names=["x", "x_len"],
        output_names=["enc_states", "enc_len", "enc_mask"],
        opset_version=14,
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_len": {0: "N"},
            "enc_states": {0: "N", 1: "T"},
            "enc_len": {0: "N"},
            "enc_mask": {0: "N", 1: "T"},
        },
    )


def export_decoder(canary_model):
    decoder = DecoderWrapper(canary_model)
    decoder_input_ids = torch.tensor([[1, 0]], dtype=torch.int32)

    decoder_mems_list_0 = torch.zeros(1, 10, 1024)
    decoder_mems_list_1 = torch.zeros(1, 10, 1024)
    decoder_mems_list_2 = torch.zeros(1, 10, 1024)
    decoder_mems_list_3 = torch.zeros(1, 10, 1024)
    decoder_mems_list_4 = torch.zeros(1, 10, 1024)
    decoder_mems_list_5 = torch.zeros(1, 10, 1024)

    enc_states = torch.zeros(1, 1000, 1024)
    enc_mask = torch.ones(1, 1000).bool()

    torch.onnx.export(
        decoder,
        (
            decoder_input_ids,
            decoder_mems_list_0,
            decoder_mems_list_1,
            decoder_mems_list_2,
            decoder_mems_list_3,
            decoder_mems_list_4,
            decoder_mems_list_5,
            enc_states,
            enc_mask,
        ),
        "decoder.onnx",
        opset_version=14,
        dynamo=False,
        input_names=[
            "decoder_input_ids",
            "decoder_mems_list_0",
            "decoder_mems_list_1",
            "decoder_mems_list_2",
            "decoder_mems_list_3",
            "decoder_mems_list_4",
            "decoder_mems_list_5",
            "enc_states",
            "enc_mask",
        ],
        output_names=[
            "logits",
            "next_decoder_mem_list_0",
            "next_decoder_mem_list_1",
            "next_decoder_mem_list_2",
            "next_decoder_mem_list_3",
            "next_decoder_mem_list_4",
            "next_decoder_mem_list_5",
        ],
        dynamic_axes={
            "decoder_mems_list_0": {1: "num_tokens"},
            "decoder_mems_list_1": {1: "num_tokens"},
            "decoder_mems_list_2": {1: "num_tokens"},
            "decoder_mems_list_3": {1: "num_tokens"},
            "decoder_mems_list_4": {1: "num_tokens"},
            "decoder_mems_list_5": {1: "num_tokens"},
            "enc_states": {1: "T"},
            "enc_mask": {1: "T"},
        },
    )


def export_tokens(canary_model):
    underline = "▁"
    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i in range(canary_model.tokenizer.vocab_size):
            s = canary_model.tokenizer.ids_to_text([i])

            if s[0] == " ":
                s = underline + s[1:]

            f.write(f"{s} {i}\n")
        print("Saved to tokens.txt")


@torch.no_grad()
def main():
    canary_model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-flash")
    canary_model = canary_model.cpu()
    canary_model.eval()

    preprocessor = canary_model.cfg["preprocessor"]
    sample_rate = preprocessor["sample_rate"]
    normalize_type = preprocessor["normalize"]
    window_size = preprocessor["window_size"]  # ms
    window_stride = preprocessor["window_stride"]  # ms
    window = preprocessor["window"]
    features = preprocessor["features"]
    n_fft = preprocessor["n_fft"]
    vocab_size = canary_model.tokenizer.vocab_size  # 5248

    subsampling_factor = canary_model.cfg["encoder"]["subsampling_factor"]

    assert sample_rate == 16000, sample_rate
    assert normalize_type == "per_feature", normalize_type
    assert window_size == 0.025, window_size
    assert window_stride == 0.01, window_stride
    assert window == "hann", window
    assert features == 128, features
    assert n_fft == 512, n_fft
    assert subsampling_factor == 8, subsampling_factor

    export_tokens(canary_model)
    export_encoder(canary_model)
    export_decoder(canary_model)

    for m in ["encoder", "decoder"]:
        quantize_dynamic(
            model_input=f"./{m}.onnx",
            model_output=f"./{m}.int8.onnx",
            weight_type=QuantType.QUInt8,
        )

    meta_data = {
        "vocab_size": vocab_size,
        "normalize_type": normalize_type,
        "subsampling_factor": subsampling_factor,
        "model_type": "EncDecMultiTaskModel",
        "version": "1",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/canary-1b-flash",
        "feat_dim": features,
    }

    add_meta_data("encoder.onnx", meta_data)
    add_meta_data("encoder.int8.onnx", meta_data)

    """
    To fix the following error with onnxruntime 1.17.1 and 1.16.3:

    onnxruntime.capi.onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 :FAIL : Load model from ./decoder.int8.onnx failed:/Users/runner/work/1/s/onnxruntime/core/graph/model.cc:150 onnxruntime::Model::Model(onnx::ModelProto &&, const onnxruntime::PathString &, const onnxruntime::IOnnxRuntimeOpSchemaRegistryList *, const logging::Logger &, const onnxruntime::ModelOptions &)
    Unsupported model IR version: 10, max supported IR version: 9
    """
    for filename in [
        "./encoder.onnx",
        "./encoder.int8.onnx",
        "./decoder.onnx",
        "./decoder.int8.onnx",
    ]:
        model = onnx.load(filename, load_external_data=False)
        print("old", model.ir_version)
        model.ir_version = 9
        print("new", model.ir_version)
        onnx.save(model, filename)

subprocess.run(["ls", "-lh", "*.onnx"], check=True)


if __name__ == "__main__":
    main()
