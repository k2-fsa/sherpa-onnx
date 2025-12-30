#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import gigaam
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

"""
NodeArg(name='features', type='tensor(float)', shape=['batch_size', 64, 'seq_len'])
NodeArg(name='feature_lengths', type='tensor(int64)', shape=['batch_size'])
-----
NodeArg(name='log_probs', type='tensor(float)', shape=['batch_size', 'seq_len', 34])
"""


def add_meta_data(filename: str, meta_data: dict[str, str]):
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


"""
{'model_class': 'ctc', 'sample_rate': 16000,
'preprocessor': {'_target_': 'gigaam.preprocess.FeatureExtractor', 'sample_rate': 16000, 'features': 64,
'win_length': 320, 'hop_length': 160, 'mel_scale': 'htk', 'n_fft': 320, 'mel_norm': None, 'center': False},
'encoder': {'_target_': 'gigaam.encoder.ConformerEncoder', 'feat_in': 64, 'n_layers': 16, 'd_model': 768,
'subsampling': 'conv1d', 'subs_kernel_size': 5, 'subsampling_factor': 4, 'ff_expansion_factor': 4,
'self_attention_model': 'rotary', 'pos_emb_max_len': 5000, 'n_heads': 16, 'conv_kernel_size': 5,
'flash_attn': False, 'conv_norm_type': 'layer_norm'}, 'head': {'_target_': 'gigaam.decoder.CTCHead',
'feat_in': 768, 'num_classes': 34}, 'decoding': {'_target_': 'gigaam.decoding.CTCGreedyDecoding',
'vocabulary': [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с',
'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']}, 'model_name': 'v3_ctc',
'hashes': {'model': '1bdc12052560591b7cdf35bef02619fa'}}
"""


def main() -> None:
    model_name = "v3_ctc"
    model = gigaam.load_model(model_name)

    # use characters
    # space is 0
    # <blk> is the last token
    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(model.cfg["decoding"]["vocabulary"]):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")
    model.to_onnx(".")
    meta_data = {
        "vocab_size": len(model.cfg["decoding"]["vocabulary"]) + 1,
        "normalize_type": "",
        "subsampling_factor": 4,
        "model_type": "EncDecCTCModel",
        "version": "1",
        "model_author": "https://github.com/salute-developers/GigaAM",
        "license": "https://github.com/salute-developers/GigaAM/blob/main/LICENSE",
        "language": "Russian",
        "comment": "v3",
        "is_giga_am": 1,
    }
    add_meta_data(f"./{model_name}.onnx", meta_data)
    quantize_dynamic(
        model_input=f"./{model_name}.onnx",
        model_output="./model.int8.onnx",
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    main()
