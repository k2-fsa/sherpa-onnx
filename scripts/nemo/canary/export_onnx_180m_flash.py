#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import os
from typing import Tuple

import nemo
import onnxmltools
import torch
from nemo.collections.common.parts import NEG_INF
from onnxmltools.utils.float16_converter import convert_float_to_float16
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


def export_onnx_fp16(onnx_fp32_path, onnx_fp16_path):
    onnx_fp32_model = onnxmltools.utils.load_model(onnx_fp32_path)
    onnx_fp16_model = convert_float_to_float16(onnx_fp32_model, keep_io_types=True)
    onnxmltools.utils.save_model(onnx_fp16_model, onnx_fp16_path)


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

        decoder_mems_list = self.decoder.decoder.forward(
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
        logits = self.log_softmax(hidden_states=decoder_mems_list[-1][:, -1:])

        return logits, decoder_mems_list


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
        dynamo=True,
        opset_version=14,
        external_data=False,
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
            "decoder_input_ids": {1: "num_tokens"},
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
    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i in range(canary_model.tokenizer.vocab_size):
            s = canary_model.tokenizer.ids_to_text([i])
            f.write(f"{s} {i}\n")
        print("Saved to tokens.txt")


@torch.no_grad()
def main():
    canary_model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-180m-flash")
    export_tokens(canary_model)
    export_encoder(canary_model)
    export_decoder(canary_model)

    for m in ["encoder", "decoder"]:
        quantize_dynamic(
            model_input=f"./{m}.onnx",
            model_output=f"./{m}.int8.onnx",
            weight_type=QuantType.QUInt8,
        )

        export_onnx_fp16(f"{m}.onnx", f"{m}.fp16.onnx")

    os.system("ls -lh *.onnx")


if __name__ == "__main__":
    main()
