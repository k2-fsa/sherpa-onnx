#!/usr/bin/env python3
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
        choices=["80", "480", "1040"],
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
    model_name = f"stt_en_fastconformer_hybrid_large_streaming_{args.model}ms"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    decoder_type = "ctc"
    asr_model.change_decoding_strategy(decoder_type=decoder_type)
    asr_model.eval()

    assert asr_model.encoder.streaming_cfg is not None
    if isinstance(asr_model.encoder.streaming_cfg.chunk_size, list):
        chunk_size = asr_model.encoder.streaming_cfg.chunk_size[1]
    else:
        chunk_size = asr_model.encoder.streaming_cfg.chunk_size

    if isinstance(asr_model.encoder.streaming_cfg.pre_encode_cache_size, list):
        pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
    else:
        pre_encode_cache_size = asr_model.encoder.streaming_cfg.pre_encode_cache_size
    window_size = chunk_size + pre_encode_cache_size

    print("chunk_size", chunk_size)
    print("pre_encode_cache_size", pre_encode_cache_size)
    print("window_size", window_size)

    chunk_shift = chunk_size

    # cache_last_channel: (batch_size, dim1, dim2, dim3)
    cache_last_channel_dim1 = len(asr_model.encoder.layers)
    cache_last_channel_dim2 = asr_model.encoder.streaming_cfg.last_channel_cache_size
    cache_last_channel_dim3 = asr_model.encoder.d_model

    # cache_last_time: (batch_size, dim1, dim2, dim3)
    cache_last_time_dim1 = len(asr_model.encoder.layers)
    cache_last_time_dim2 = asr_model.encoder.d_model
    cache_last_time_dim3 = asr_model.encoder.conv_context_size[0]

    asr_model.set_export_config({"decoder_type": "ctc", "cache_support": True})

    filename = "model.onnx"

    asr_model.export(filename)

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "window_size": window_size,
        "chunk_shift": chunk_shift,
        "normalize_type": "None",
        "cache_last_channel_dim1": cache_last_channel_dim1,
        "cache_last_channel_dim2": cache_last_channel_dim2,
        "cache_last_channel_dim3": cache_last_channel_dim3,
        "cache_last_time_dim1": cache_last_time_dim1,
        "cache_last_time_dim2": cache_last_time_dim2,
        "cache_last_time_dim3": cache_last_time_dim3,
        "subsampling_factor": 8,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": f"https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/{model_name}",
        "comment": "Only the CTC branch is exported",
    }
    add_meta_data(filename, meta_data)

    print(meta_data)


if __name__ == "__main__":
    main()
