#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import os
from typing import Any, Dict, List, Tuple

import onnx
import sentencepiece as spm
import torch

from torch_model import SenseVoiceSmall


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-len-in-seconds",
        type=int,
        required=True,
        help="""RKNN does not support dynamic shape, so we need to hard-code
        how long the model can process.
        """,
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
    )
    return parser.parse_args()


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


def load_cmvn(filename) -> Tuple[List[float], List[float]]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = list(map(lambda x: float(x), t))
            else:
                inv_stddev = list(map(lambda x: float(x), t))

    return neg_mean, inv_stddev


def generate_tokens(sp):
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    print("saved to tokens.txt")


@torch.no_grad()
def main():
    args = get_args()
    print(vars(args))

    sp = spm.SentencePieceProcessor()
    sp.load("./chn_jpn_yue_eng_ko_spectok.bpe.model")
    vocab_size = sp.vocab_size()
    generate_tokens(sp)

    print("loading model")

    state_dict = torch.load("./model.pt", map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    neg_mean, inv_stddev = load_cmvn("./am.mvn")

    neg_mean = torch.tensor(neg_mean, dtype=torch.float32)
    inv_stddev = torch.tensor(inv_stddev, dtype=torch.float32)

    model = SenseVoiceSmall(neg_mean=neg_mean, inv_stddev=inv_stddev)
    model.load_state_dict(state_dict)
    model.eval()
    del state_dict

    lfr_window_size = 7
    lfr_window_shift = 6

    # frame shift is 10ms, 1 second has about 100 feature frames
    input_len_in_seconds = int(args.input_len_in_seconds)
    num_frames = input_len_in_seconds * 100
    print("num_frames", num_frames)

    # num_input_frames is an approximate number
    num_input_frames = int(num_frames / lfr_window_shift + 0.5)
    print("num_input_frames", num_input_frames)

    x = torch.randn(1, num_input_frames, 560, dtype=torch.float32)

    language = 3
    text_norm = 15
    prompt = torch.tensor([language, 1, 2, text_norm], dtype=torch.int32)

    opset_version = args.opset_version
    filename = f"model-{input_len_in_seconds}-seconds.onnx"
    torch.onnx.export(
        model,
        (x, prompt),
        filename,
        opset_version=opset_version,
        input_names=["x", "prompt"],
        output_names=["logits"],
        dynamic_axes={},
    )

    model_author = os.environ.get("model_author", "iic")
    comment = os.environ.get("comment", "iic/SenseVoiceSmall")
    url = os.environ.get("url", "https://huggingface.co/FunAudioLLM/SenseVoiceSmall")

    meta_data = {
        "lfr_window_size": lfr_window_size,
        "lfr_window_shift": lfr_window_shift,
        "num_input_frames": num_input_frames,
        "normalize_samples": 0,  # input should be in the range [-32768, 32767]
        "model_type": "sense_voice_ctc",
        "version": "1",
        "model_author": model_author,
        "maintainer": "k2-fsa",
        "vocab_size": vocab_size,
        "comment": comment,
        "lang_auto": model.lid_dict["auto"],
        "lang_zh": model.lid_dict["zh"],
        "lang_en": model.lid_dict["en"],
        "lang_yue": model.lid_dict["yue"],  # cantonese
        "lang_ja": model.lid_dict["ja"],
        "lang_ko": model.lid_dict["ko"],
        "lang_nospeech": model.lid_dict["nospeech"],
        "with_itn": model.textnorm_dict["withitn"],
        "without_itn": model.textnorm_dict["woitn"],
        "url": url,
    }
    add_meta_data(filename=filename, meta_data=meta_data)


if __name__ == "__main__":
    torch.manual_seed(20250717)
    main()
