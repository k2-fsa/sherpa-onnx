#!/usr/bin/env python3
# Copyright (c)  2026  graehl

"""
Export a w2v-BERT 2.0 CTC model (transformers' Wav2Vec2BertForCTC) to ONNX.

Usage:

    ./export_onnx.py --model-id proxectonos/w2v-bert-2.0-gl

It writes to the current directory:

  - model.onnx, model.weights: fp32. The weights exceed protobuf's 2 GB
    limit, so they are a separate file, which must stay beside model.onnx.
  - model.int8.onnx: the MatMul weights dynamically quantized to uint8, in
    one file.
  - tokens.txt

The graph takes

  - input_features: float32 (N, T, 160), what the checkpoint's
    SeamlessM4TFeatureExtractor computes: 80 Kaldi filterbank energies
    per 10 ms, each bin normalized to zero mean and unit variance over the
    utterance, and each two consecutive frames joined into one row of 160.
    See compute_features() in test_onnx.py.
  - input_lengths: int64 (N,), the valid rows of each utterance.

and returns

  - logits: float32 (N, T_out, vocab_size), log-softmax normalized (log
    probabilities; version 1 exports were not normalized)
  - logits_lengths: int64 (N,), the valid output frames of each utterance.

Padding is masked inside the graph, so a zero-padded batch gives each
utterance the logits it gets alone.
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForCTC, AutoProcessor


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="proxectonos/w2v-bert-2.0-gl",
        help="A Wav2Vec2BertForCTC checkpoint on Hugging Face, or a local "
        "directory holding one",
    )
    return parser.parse_args()


def length_mask(max_len: int, lengths: torch.Tensor) -> torch.Tensor:
    """(N, max_len) booleans, True at the valid positions."""
    return torch.arange(max_len, device=lengths.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)


class Wrapper(torch.nn.Module):
    """Wav2Vec2BertForCTC with an input_lengths input.

    transformers masks padding in the encoder layers, but the subsampling
    adapter after them convolves over the padded frames, and its attention
    then spreads them to every frame. So the padded frames are zeroed at
    the input of each adapter convolution, which is what that convolution's
    own zero padding gives an utterance run alone.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.adapter_lengths = []  # valid frames entering each adapter layer
        adapter = m.wav2vec2_bert.adapter
        for i, layer in enumerate(adapter.layers if adapter is not None else []):
            for conv in (layer.residual_conv, layer.self_attn_conv):
                conv.register_forward_pre_hook(
                    lambda _, args, i=i: (self.zero_padding(args[0], i),)
                )

    def zero_padding(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """x is (N, C, T)."""
        valid = length_mask(x.shape[2], self.adapter_lengths[layer])
        return x * valid.unsqueeze(1).to(x.dtype)

    def forward(self, input_features: torch.Tensor, input_lengths: torch.Tensor):
        """
        Args:
          input_features: (N, T, 160), float32
          input_lengths: (N,), int64
        Returns:
          logits: (N, T_out, vocab_size), float32, log-softmax normalized
          logits_lengths: (N,), int64
        """
        adapter = self.m.wav2vec2_bert.adapter
        lengths = input_lengths
        self.adapter_lengths = []
        for layer in adapter.layers if adapter is not None else []:
            self.adapter_lengths.append(lengths)
            conv = layer.residual_conv
            lengths = (
                torch.div(
                    lengths + 2 * conv.padding[0] - conv.kernel_size[0],
                    conv.stride[0],
                    rounding_mode="floor",
                )
                + 1
            )
        mask = length_mask(input_features.shape[1], input_lengths)
        logits = self.m(input_features=input_features, attention_mask=mask).logits
        # Normalized in the graph, so the runtime need not.
        return logits.log_softmax(dim=-1), lengths.to(torch.int64)


def subsampling_factor(m) -> int:
    """Input filterbank frames (10 ms) per output frame."""
    factor = 2  # the feature extractor joins two frames into one row
    adapter = m.wav2vec2_bert.adapter
    for layer in adapter.layers if adapter is not None else []:
        factor *= layer.residual_conv.stride[0]
    return factor


def generate_tokens(tokenizer, vocab_size: int, blank_id: int):
    """sherpa-onnx reads a leading U+2581 as the start of a word, finds the
    CTC blank by the name <blk>, and does not output a symbol in angle
    brackets, so the word delimiter, the padding token (the blank) and the
    unknown token are renamed."""
    id2token = {i: t for t, i in tokenizer.get_vocab().items()}
    id2token[tokenizer.convert_tokens_to_ids(tokenizer.word_delimiter_token)] = "▁"
    id2token[blank_id] = "<blk>"
    id2token[tokenizer.unk_token_id] = "<unk>"
    missing = [i for i in range(vocab_size) if i not in id2token]
    if missing:
        raise ValueError(f"The vocabulary has no token for IDs {missing}")
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(vocab_size):
            t = id2token[i]
            if not t or " " in t:
                raise ValueError(f"Token {i} {t!r} is empty or contains a space")
            f.write(f"{t} {i}\n")
    print("saved to tokens.txt")


def save_with_meta_data(src: str, filename: str, meta_data: Dict[str, Any]):
    """Save the graph at src as filename with meta data, its weights in one
    file, model.weights."""
    model = onnx.load(src)
    model.metadata_props.clear()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    Path("model.weights").unlink(missing_ok=True)
    onnx.save(
        model,
        filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.weights",
    )


@torch.no_grad()
def main():
    args = get_args()
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForCTC.from_pretrained(args.model_id, dtype=torch.float32)
    model.eval()

    vocab_size = model.config.vocab_size
    blank_id = model.config.pad_token_id
    generate_tokens(processor.tokenizer, vocab_size, blank_id)

    w = Wrapper(model)
    w.eval()

    x = torch.rand(2, 120, 160)
    x_lengths = torch.tensor([120, 90], dtype=torch.int64)

    # The exporter writes weights over 2 GB as one file per tensor beside
    # the graph, so it writes into its own directory and
    # save_with_meta_data() saves the graph again with one weights file.
    tmp = Path("export.tmp")
    tmp.mkdir(exist_ok=True)
    torch.onnx.export(
        w,
        (x, x_lengths),
        str(tmp / "model.onnx"),
        input_names=["input_features", "input_lengths"],
        output_names=["logits", "logits_lengths"],
        dynamic_axes={
            "input_features": {0: "N", 1: "T"},
            "input_lengths": {0: "N"},
            "logits": {0: "N", 1: "T_out"},
            "logits_lengths": {0: "N"},
        },
        opset_version=17,
        dynamo=False,
    )

    filename = "model.onnx"
    meta_data = {
        "model_type": "w2v-bert-ctc",
        "version": "2",
        "model_author": "facebook",
        "maintainer": "k2-fsa",
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "subsampling_factor": subsampling_factor(model),
        "feature_dim": 160,
        "sample_rate": 16000,
        "checkpoint": args.model_id,
        "url": f"https://huggingface.co/{args.model_id}",
        "comment": "input_features as SeamlessM4TFeatureExtractor computes "
        "them; padding is masked in the graph",
    }
    save_with_meta_data(str(tmp / "model.onnx"), filename, meta_data)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()

    quantize_dynamic(
        model_input=filename,
        model_output="model.int8.onnx",
        op_types_to_quantize=["MatMul"],
        # As for the other models in this repository: QInt8 gives wrong
        # results with the C++ onnxruntime.
        weight_type=QuantType.QUInt8,
    )
    print("saved to model.onnx, model.weights and model.int8.onnx")


if __name__ == "__main__":
    main()
