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
import sentencepiece as spm
import torch
from torch_model import SenseVoiceSmall
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


def generate_tokens(sp):
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    print("saved to tokens.txt")


@torch.no_grad()
def main():
    sp = spm.SentencePieceProcessor()
    sp.load("./chn_jpn_yue_eng_ko_spectok.bpe.model")
    vocab_size = sp.vocab_size()
    generate_tokens(sp)

    print("loading model")

    state_dict = torch.load("./model.pt")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model = SenseVoiceSmall()
    model.load_state_dict(state_dict)
    del state_dict

    x = torch.randn(1, 100, 560, dtype=torch.float32)
    language = torch.tensor([3], dtype=torch.int32)
    text_norm = torch.tensor([15], dtype=torch.int32)

    opset_version = 13
    filename = "model.onnx"
    torch.onnx.export(
        model,
        (x, language, text_norm),
        filename,
        opset_version=opset_version,
        input_names=["x", "language", "text_norm"],
        output_names=["logits"],
        dynamic_axes={},
    )

    lfr_window_size = 7
    lfr_window_shift = 6

    neg_mean, inv_stddev = load_cmvn("./am.mvn")

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
    torch.manual_seed(20250717)
    main()
