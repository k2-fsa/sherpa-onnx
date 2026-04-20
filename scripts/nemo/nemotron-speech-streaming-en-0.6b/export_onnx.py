#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)
from typing import Dict

import nemo.collections.asr as nemo_asr
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

"""
'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor',
'sample_rate': 16000, 'normalize': 'NA', 'window_size': 0.025, 'window_stride': 0.01,
'window': 'hann', 'features': 128, 'n_fft': 512, 'log': True,
'frame_splicing': 1, 'dither': 1e-05, 'pad_to': 0, 'pad_value': 0.0}

"""


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

    external_filename = filename.split(".onnx")[0]
    onnx.save(
        model,
        filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_filename + ".data",
    )


@torch.no_grad()
def main():
    model_name = "nvidia/nemotron-speech-streaming-en-0.6b"

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

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

    asr_model.set_export_config({"cache_support": True})

    asr_model.encoder.export("encoder.onnx")
    asr_model.decoder.export("decoder.onnx")
    asr_model.joint.export("joiner.onnx")

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "window_size": window_size,
        "chunk_shift": chunk_shift,
        "normalize_type": normalize_type,
        "cache_last_channel_dim1": cache_last_channel_dim1,
        "cache_last_channel_dim2": cache_last_channel_dim2,
        "cache_last_channel_dim3": cache_last_channel_dim3,
        "cache_last_time_dim1": cache_last_time_dim1,
        "cache_last_time_dim2": cache_last_time_dim2,
        "cache_last_time_dim3": cache_last_time_dim3,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "feat_dim": 128,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "version": "1",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b",
        "comment": "Only the transducer branch is exported",
    }
    add_meta_data("encoder.onnx", meta_data)

    for m in ["encoder", "decoder", "joiner"]:
        quantize_dynamic(
            model_input=f"{m}.onnx",
            model_output=f"{m}.int8.onnx",
            weight_type=QuantType.QUInt8,
        )

    print(meta_data)


if __name__ == "__main__":
    main()
