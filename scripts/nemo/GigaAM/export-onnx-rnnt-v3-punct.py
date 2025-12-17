#!/usr/bin/env python3
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)
import os

import gigaam
import onnx
import torch
from gigaam.utils import onnx_converter
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor

# encoder input length should be of int64
# encder output length can be int64 or int32

"""
==========Input==========
NodeArg(name='audio_signal', type='tensor(float)', shape=['batch_size', 64, 'seq_len'])
NodeArg(name='length', type='tensor(int64)', shape=['batch_size'])
==========Output==========
NodeArg(name='encoded', type='tensor(float)', shape=['batch_size', 768, 'Transposeencoded_dim_2'])
NodeArg(name='encoded_len', type='tensor(int32)', shape=['batch_size'])
==========Input==========
NodeArg(name='x', type='tensor(int32)', shape=[1, 1])
NodeArg(name='unused_x_len.1', type='tensor(int32)', shape=[1])
NodeArg(name='h.1', type='tensor(float)', shape=[1, 1, 320])
NodeArg(name='c.1', type='tensor(float)', shape=[1, 1, 320])
==========Output==========
NodeArg(name='dec', type='tensor(float)', shape=[1, 320, 1])
NodeArg(name='unused_x_len', type='tensor(int32)', shape=[1])
NodeArg(name='h', type='tensor(float)', shape=[1, 1, 320])
NodeArg(name='c', type='tensor(float)', shape=[1, 1, 320])
==========Input==========
NodeArg(name='enc', type='tensor(float)', shape=[1, 768, 1])
NodeArg(name='dec', type='tensor(float)', shape=[1, 320, 1])
==========Output==========
NodeArg(name='joint', type='tensor(float)', shape=[1, 1, 1, 1025])
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


class EncoderWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, audio_signal: Tensor, length: Tensor):
        # https://github.com/salute-developers/GigaAM/blob/main/gigaam/encoder.py#L499
        out, out_len = self.m.encoder(audio_signal, length)

        return out, out_len.to(torch.int64)

    def to_onnx(self, dir_path: str = "."):
        onnx_converter(
            model_name=f"{self.m.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.m.encoder,
            dynamic_axes=self.m.encoder.dynamic_axes(),
        )


class DecoderWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x: Tensor, unused_x_len: Tensor, h: Tensor, c: Tensor):
        # https://github.com/salute-developers/GigaAM/blob/main/gigaam/decoder.py#L110C17-L110C54
        emb = self.m.head.decoder.embed(x)
        g, (h, c) = self.m.head.decoder.lstm(emb.transpose(0, 1), (h, c))
        return g.permute(1, 2, 0), unused_x_len + 1, h, c

    def to_onnx(self, dir_path: str = "."):
        label, hidden_h, hidden_c = self.m.head.decoder.input_example()
        label = label.to(torch.int32)
        label_len = torch.zeros(1, dtype=torch.int32)

        onnx_converter(
            model_name=f"{self.m.cfg.model_name}_decoder",
            out_dir=dir_path,
            module=self,
            dynamic_axes=self.m.encoder.dynamic_axes(),
            inputs=(label, label_len, hidden_h, hidden_c),
            input_names=["x", "unused_x_len.1", "h.1", "c.1"],
            output_names=["dec", "unused_x_len", "h", "c"],
        )


"""
{'model_class': 'rnnt', 'sample_rate': 16000,
'preprocessor': {'_target_': 'gigaam.preprocess.FeatureExtractor', 'sample_rate': 16000,
'features': 64, 'win_length': 320, 'hop_length': 160, 'mel_scale': 'htk', 'n_fft': 320,
'mel_norm': None, 'center': False},
'encoder': {'_target_': 'gigaam.encoder.ConformerEncoder', 'feat_in': 64, 'n_layers': 16,
'd_model': 768, 'subsampling_factor': 4, 'ff_expansion_factor': 4,
'self_attention_model': 'rotary', 'pos_emb_max_len': 5000, 'n_heads': 16,
'conv_kernel_size': 5, 'flash_attn': False, 'subs_kernel_size': 5,
'subsampling': 'conv1d', 'conv_norm_type': 'layer_norm'},
'head': {'_target_': 'gigaam.decoder.RNNTHead',
'decoder': {'pred_hidden': 320, 'pred_rnn_layers': 1, 'num_classes': 1025},
'joint': {'enc_hidden': 768, 'pred_hidden': 320, 'joint_hidden': 320, 'num_classes': 1025}},
'decoding': {'_target_': 'gigaam.decoding.RNNTGreedyDecoding',
'vocabulary': None, 'model_path': '/root/.cache/gigaam/v3_e2e_rnnt_tokenizer.model'}, 'model_name': 'v3_e2e_rnnt', 'hashes': {'model': '72e2a9b5c7caad963b2bbfd2f298c252', 'tokenizer': '3b3bf8370e882885d79731592fc99f98'}}
"""


def main() -> None:
    model_name = "v3_e2e_rnnt"
    model = gigaam.load_model(model_name)

    # <blk> is the last token
    sp = model.decoding.tokenizer.model
    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")

        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    EncoderWrapper(model).to_onnx(".")
    DecoderWrapper(model).to_onnx(".")

    onnx_converter(
        model_name=f"{model.cfg.model_name}_joint",
        out_dir=".",
        module=model.head.joint,
    )
    meta_data = {
        # vocab_size does not include the blank
        # we will increase vocab_size by 1 in the c++ code
        "vocab_size": model.cfg["head"]["decoder"]["num_classes"] - 1,
        "pred_rnn_layers": model.cfg["head"]["decoder"]["pred_rnn_layers"],
        "pred_hidden": model.cfg["head"]["decoder"]["pred_hidden"],
        "normalize_type": "",
        "subsampling_factor": 4,
        "model_type": "EncDecRNNTBPEModel",
        "version": "3",
        "model_author": "https://github.com/salute-developers/GigaAM",
        "license": "https://github.com/salute-developers/GigaAM/blob/main/LICENSE",
        "language": "Russian",
        "comment": "v3",
        "is_giga_am": 1,
    }

    add_meta_data(f"./{model_name}_encoder.onnx", meta_data)
    quantize_dynamic(
        model_input=f"./{model_name}_encoder.onnx",
        model_output="./encoder.int8.onnx",
        weight_type=QuantType.QUInt8,
    )
    os.rename(f"./{model_name}_decoder.onnx", "decoder.onnx")
    os.rename(f"./{model_name}_joint.onnx", "joiner.onnx")
    os.remove(f"./{model_name}_encoder.onnx")


if __name__ == "__main__":
    main()
