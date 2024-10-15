#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

"""
We use
https://hf-mirror.com/yuekai/model_repo_sense_voice_small/blob/main/export_onnx.py
as a reference while writing this file.

Thanks to https://github.com/yuekaizhang for making the file public.
"""

import os
from typing import Any, Dict, Tuple

import onnx
import torch
from model import SenseVoiceSmall
from onnxruntime.quantization import QuantType, quantize_dynamic


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


def modified_forward(
    self,
    x: torch.Tensor,
    x_length: torch.Tensor,
    language: torch.Tensor,
    text_norm: torch.Tensor,
):
    """
    Args:
      x:
        A 3-D tensor of shape (N, T, C) with dtype torch.float32
      x_length:
        A 1-D tensor of shape (N,) with dtype torch.int32
      language:
        A 1-D tensor of shape (N,) with dtype torch.int32
        See also https://github.com/FunAudioLLM/SenseVoice/blob/a80e676461b24419cf1130a33d4dd2f04053e5cc/model.py#L640
      text_norm:
        A 1-D tensor of shape (N,) with dtype torch.int32
        See also https://github.com/FunAudioLLM/SenseVoice/blob/a80e676461b24419cf1130a33d4dd2f04053e5cc/model.py#L642
    """
    language_query = self.embed(language).unsqueeze(1)
    text_norm_query = self.embed(text_norm).unsqueeze(1)

    event_emo_query = self.embed(torch.LongTensor([[1, 2]])).repeat(x.size(0), 1, 1)

    x = torch.cat((language_query, event_emo_query, text_norm_query, x), dim=1)
    x_length += 4

    encoder_out, encoder_out_lens = self.encoder(x, x_length)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    ctc_logits = self.ctc.ctc_lo(encoder_out)

    return ctc_logits


def load_cmvn(filename) -> Tuple[str, str]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def generate_tokens(params):
    sp = params["tokenizer"].sp
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")

    os.system("head tokens.txt; tail -n200 tokens.txt")


def display_params(params):
    print("----------params----------")
    print(params)

    print("----------frontend_conf----------")
    print(params["frontend_conf"])

    os.system(f"cat {params['frontend_conf']['cmvn_file']}")

    print("----------config----------")
    print(params["config"])

    os.system(f"cat {params['config']}")


def main():
    model, params = SenseVoiceSmall.from_pretrained(model="iic/SenseVoiceSmall")
    display_params(params)

    generate_tokens(params)

    model.__class__.forward = modified_forward

    x = torch.randn(2, 100, 560, dtype=torch.float32)
    x_length = torch.tensor([80, 100], dtype=torch.int32)
    language = torch.tensor([0, 3], dtype=torch.int32)
    text_norm = torch.tensor([14, 15], dtype=torch.int32)

    opset_version = 13
    filename = "model.onnx"
    torch.onnx.export(
        model,
        (x, x_length, language, text_norm),
        filename,
        opset_version=opset_version,
        input_names=["x", "x_length", "language", "text_norm"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "N", 1: "T"},
            "x_length": {0: "N"},
            "language": {0: "N"},
            "text_norm": {0: "N"},
            "logits": {0: "N", 1: "T"},
        },
    )

    lfr_window_size = params["frontend_conf"]["lfr_m"]
    lfr_window_shift = params["frontend_conf"]["lfr_n"]

    neg_mean, inv_stddev = load_cmvn(params["frontend_conf"]["cmvn_file"])
    vocab_size = params["tokenizer"].sp.vocab_size()

    meta_data = {
        "lfr_window_size": lfr_window_size,
        "lfr_window_shift": lfr_window_shift,
        "normalize_samples": 0,  # input should be in the range [-32768, 32767]
        "neg_mean": neg_mean,
        "inv_stddev": inv_stddev,
        "model_type": "sense_voice_ctc",
        # version 1: Use QInt8
        # version 2: Use QUInt8
        "version": "2",
        "model_author": "iic",
        "maintainer": "k2-fsa",
        "vocab_size": vocab_size,
        "comment": "iic/SenseVoiceSmall",
        "lang_auto": model.lid_dict["auto"],
        "lang_zh": model.lid_dict["zh"],
        "lang_en": model.lid_dict["en"],
        "lang_yue": model.lid_dict["yue"],  # cantonese
        "lang_ja": model.lid_dict["ja"],
        "lang_ko": model.lid_dict["ko"],
        "lang_nospeech": model.lid_dict["nospeech"],
        "with_itn": model.textnorm_dict["withitn"],
        "without_itn": model.textnorm_dict["woitn"],
        "url": "https://huggingface.co/FunAudioLLM/SenseVoiceSmall",
    }
    add_meta_data(filename=filename, meta_data=meta_data)

    filename_int8 = "model.int8.onnx"
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        op_types_to_quantize=["MatMul"],
        # Note that we have to use QUInt8 here.
        #
        # When QInt8 is used, C++ onnxruntime produces incorrect results
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    torch.manual_seed(20240717)
    main()
