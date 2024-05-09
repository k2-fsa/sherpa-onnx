#!/usr/bin/env python3
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)
import argparse
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--doc",
        type=str,
        default="",
    )
    return parser.parse_args()


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

    onnx.save(model, filename)


@torch.no_grad()
def main():
    args = get_args()
    model_name = args.model

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    print(asr_model.cfg)
    print(asr_model)

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    decoder_type = "ctc"
    asr_model.change_decoding_strategy(decoder_type=decoder_type)
    asr_model.eval()

    asr_model.set_export_config({"decoder_type": "ctc"})

    filename = "model.onnx"

    asr_model.export(filename)

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "subsampling_factor": 8,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/{model_name}",
        "comment": "Only the CTC branch is exported",
        "doc": args.doc,
    }
    add_meta_data(filename, meta_data)

    print("preprocessor", asr_model.cfg.preprocessor)
    print(meta_data)


if __name__ == "__main__":
    main()
