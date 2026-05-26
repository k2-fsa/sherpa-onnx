#!/usr/bin/env python3
# Copyright      2026  Milan Leonard
"""Buffered streaming ONNX export for nvidia/parakeet-unified-en-0.6b."""

import argparse
from pathlib import Path
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

LATENCY_PRESETS = {
    "1120ms": {"left": 70, "chunk": 7, "right": 7},
    "560ms": {"left": 70, "chunk": 2, "right": 5},
    "240ms": {"left": 70, "chunk": 1, "right": 2},
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Buffered streaming ONNX export for parakeet-unified-en-0.6b",
    )
    parser.add_argument(
        "--latency",
        type=str,
        default="1120ms",
        choices=sorted(LATENCY_PRESETS.keys()),
        help="Latency preset to export.",
    )
    return parser.parse_args()


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    if Path(filename).name == "encoder.onnx":
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="encoder.weights",
        )
    else:
        onnx.save(model, filename)


def print_onnx_listing():
    for p in sorted(Path.cwd().glob("*.onnx")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"{size_mb:8.2f} MB  {p.name}")


@torch.no_grad()
def main():
    args = get_args()
    preset = LATENCY_PRESETS[args.latency]

    if Path("./parakeet-unified-en-0.6b.nemo").is_file():
        asr_model = nemo_asr.models.ASRModel.restore_from(
            restore_path="./parakeet-unified-en-0.6b.nemo"
        )
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="nvidia/parakeet-unified-en-0.6b"
        )

    asr_model.eval()
    asr_model.cfg.validation_ds = dict()
    asr_model.encoder.set_default_att_context_size(
        [preset["left"], preset["chunk"], preset["right"]]
    )

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i + 1}\n")
        print("Saved to tokens.txt")

    asr_model.encoder.export("encoder.onnx")
    asr_model.decoder.export("decoder.onnx")
    asr_model.joint.export("joiner.onnx")
    print_onnx_listing()

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    subsampling_factor = asr_model.encoder.subsampling_factor

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": subsampling_factor,
        "model_type": "EncDecRNNTBPEModel",
        "streaming_model_type": "nemo_parakeet_unified_streaming",
        "buffered_streaming": 1,
        "left_encoder_frames": preset["left"],
        "chunk_encoder_frames": preset["chunk"],
        "right_encoder_frames": preset["right"],
        "left_feature_frames": preset["left"] * subsampling_factor,
        "chunk_feature_frames": preset["chunk"] * subsampling_factor,
        "right_feature_frames": preset["right"] * subsampling_factor,
        "version": "2",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/parakeet-unified-en-0.6b",
        "comment": f"Buffered streaming export, latency={args.latency}",
        "feat_dim": 128,
        "latency": args.latency,
    }

    for m in ["encoder", "decoder", "joiner"]:
        quantize_dynamic(
            model_input=f"./{m}.onnx",
            model_output=f"./{m}.int8.onnx",
            weight_type=QuantType.QUInt8 if m == "encoder" else QuantType.QInt8,
        )
        print_onnx_listing()

    add_meta_data("encoder.onnx", meta_data)
    add_meta_data("encoder.int8.onnx", meta_data)
    decoder_meta_data = {
        "streaming_model_type": "nemo_parakeet_unified_streaming",
    }
    add_meta_data("decoder.onnx", decoder_meta_data)
    add_meta_data("decoder.int8.onnx", decoder_meta_data)
    print("meta_data", meta_data)


if __name__ == "__main__":
    main()
