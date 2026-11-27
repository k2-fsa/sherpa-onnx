#!/usr/bin/env python3
# Copyright (c)  2026  Anthony Mercer
"""
Export https://huggingface.co/third-intelligence/Haas-ja-v1.0 to ONNX.

Haas-ja-v1.0 is a Parakeet-TDT model (FastConformer encoder, TDT joint) whose
prediction network replaces the LSTM with two pre-norm residual layers of Kimi
Delta Attention (KDA) from flash-linear-attention. The encoder and the joiner
are exported with NeMo as for parakeet-tdt-0.6b-v2. The KDA kernels are
Triton-only and cannot be exported, so the decoder is exported from
HaasDecoder below, which runs one KDA step in plain PyTorch.

The decoder keeps the state layout of the model (see haas_asr/prednet.py):
two tensors of shape (pred_rnn_layers, N, decoder_state_dim), like the two
states of an LSTM prediction network:

  state0[i]: recurrent memory of layer i, [heads, head_v_dim, head_k_dim]
  state1[i]: short-convolution caches of layer i (q, k, v), then zeros
"""

import os
import sys
from pathlib import Path
from typing import Dict

import haas_asr  # noqa: F401, lets NeMo build the KDA decoder
import nemo.collections.asr as nemo_asr
import onnx
import torch
import torch.nn.functional as F
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

# Add parent directory to path to import generate_bpe_vocab
sys.path.insert(0, str(Path(__file__).parent.parent))
from generate_bpe_vocab import generate_bpe_vocab_from_model


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

    if filename == "encoder.onnx":
        external_filename = "encoder"
        onnx.save(
            model,
            filename,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_filename + ".weights",
        )
    else:
        onnx.save(model, filename)


class KdaStep(nn.Module):
    """One step of fla's KimiDeltaAttention, as its fused_recurrent kernel
    computes it with use_qk_l2norm_in_kernel, use_gate_in_kernel,
    use_beta_sigmoid_in_kernel and state_v_first set."""

    def __init__(self, layer):
        super().__init__()
        assert layer.num_heads == layer.num_v_heads, layer
        assert layer.use_short_conv and not layer.conv_bias, layer
        assert not layer.allow_neg_eigval and layer.lower_bound is None, layer

        self.num_heads = layer.num_heads
        self.head_k_dim = layer.head_k_dim
        self.head_v_dim = layer.head_v_dim
        self.key_dim = layer.key_dim
        self.value_dim = layer.value_dim
        self.conv_size = layer.conv_size

        self.q_proj = layer.q_proj
        self.k_proj = layer.k_proj
        self.v_proj = layer.v_proj
        self.f_proj = layer.f_proj
        self.b_proj = layer.b_proj
        self.g_proj = layer.g_proj
        self.o_proj = layer.o_proj

        # depthwise convolution weights: (dim, 1, conv_size) -> (dim, conv_size)
        self.register_buffer("q_conv", layer.q_conv1d.weight.detach()[:, 0].clone())
        self.register_buffer("k_conv", layer.k_conv1d.weight.detach()[:, 0].clone())
        self.register_buffer("v_conv", layer.v_conv1d.weight.detach()[:, 0].clone())

        a = -layer.A_log.detach().float().exp().view(self.num_heads, 1)
        self.register_buffer("neg_a", a)
        dt_bias = layer.dt_bias.detach().float()
        self.register_buffer("dt_bias", dt_bias.view(self.num_heads, self.head_k_dim))
        self.register_buffer("o_norm_weight", layer.o_norm.weight.detach().clone())
        self.o_norm_eps = layer.o_norm.eps
        self.scale = self.head_k_dim**-0.5

    @staticmethod
    def short_conv(cache, x, weight):
        # cache: (N, dim, conv_size), oldest first; x: (N, dim)
        cache = torch.cat([cache[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        return F.silu((cache * weight).sum(dim=-1)), cache

    def forward(self, x, recurrent, conv):
        """
        Args:
          x: (N, hidden), output of the layer's pre-norm
          recurrent: (N, num_heads * head_v_dim * head_k_dim)
          conv: (N, width), q, k and v caches followed by zeros
        Returns:
          (out, new_recurrent, new_conv)
        """
        n = x.shape[0]
        h, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        k_size = self.key_dim * self.conv_size
        v_size = self.value_dim * self.conv_size

        q_cache = conv[:, :k_size].reshape(n, self.key_dim, self.conv_size)
        k_cache = conv[:, k_size : 2 * k_size].reshape(n, self.key_dim, self.conv_size)
        v_cache = conv[:, 2 * k_size : 2 * k_size + v_size].reshape(
            n, self.value_dim, self.conv_size
        )

        q, q_cache = self.short_conv(q_cache, self.q_proj(x), self.q_conv)
        k, k_cache = self.short_conv(k_cache, self.k_proj(x), self.k_conv)
        v, v_cache = self.short_conv(v_cache, self.v_proj(x), self.v_conv)

        q = q.reshape(n, h, dk)
        k = k.reshape(n, h, dk)
        v = v.reshape(n, h, dv)

        q = q / torch.sqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6) * self.scale
        k = k / torch.sqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)

        # per-channel decay in log space: -exp(A_log) * softplus(g + dt_bias)
        g = self.f_proj(x).reshape(n, h, dk)
        g = self.neg_a * F.softplus(g + self.dt_bias)
        beta = torch.sigmoid(self.b_proj(x)).unsqueeze(-1)

        # delta rule with the state stored as (N, h, dv, dk)
        s = recurrent.reshape(n, h, dv, dk) * g.exp().unsqueeze(2)
        v = (v - (s * k.unsqueeze(2)).sum(dim=-1)) * beta
        s = s + v.unsqueeze(-1) * k.unsqueeze(2)
        o = (s * q.unsqueeze(2)).sum(dim=-1)

        # gated RMSNorm over each head, then the output projection
        o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + self.o_norm_eps)
        gate = self.g_proj(x).reshape(n, h, dv)
        o = o * self.o_norm_weight * torch.sigmoid(gate)
        out = self.o_proj(o.reshape(n, h * dv))

        new_conv = torch.cat(
            [
                q_cache.reshape(n, -1),
                k_cache.reshape(n, -1),
                v_cache.reshape(n, -1),
                conv[:, 2 * k_size + v_size :],
            ],
            dim=-1,
        )
        return out, s.reshape(n, -1), new_conv


class HaasDecoder(nn.Module):
    """Prediction network of Haas-ja-v1.0 for one label.

    It has the inputs and outputs of the LSTM decoders exported from NeMo, so
    it is called in the same way.
    """

    def __init__(self, decoder):
        super().__init__()
        prednet = decoder.prediction["dec_rnn"]
        self.embed = decoder.prediction["embed"]
        self.norms = prednet.norms
        self.final_norm = prednet.final_norm
        self.steps = nn.ModuleList(KdaStep(layer) for layer in prednet.layers)
        self.recurrent_size = prednet.layout.recurrent_size

    def forward(
        self,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        state0: torch.Tensor,
        state1: torch.Tensor,
    ):
        """
        Args:
          targets: (N, 1), int32
          target_length: (N,), int32
          state0: (pred_rnn_layers, N, decoder_state_dim)
          state1: (pred_rnn_layers, N, decoder_state_dim)
        Returns:
          decoder_out: (N, pred_hidden, 1)
          decoder_out_length: (N,)
          next_state0, next_state1: same shapes as state0 and state1
        """
        # The blank is the padding index of the embedding, so it gives the
        # zero input that NeMo uses for the start of sequence.
        x = self.embed(targets[:, 0].long())

        next_state0 = []
        next_state1 = []
        for i, (norm, step) in enumerate(zip(self.norms, self.steps)):
            out, recurrent, conv = step(
                norm(x), state0[i, :, : self.recurrent_size], state1[i]
            )
            x = x + out
            next_state0.append(
                torch.cat([recurrent, state0[i, :, self.recurrent_size :]], dim=-1)
            )
            next_state1.append(conv)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return (
            x.unsqueeze(-1),
            target_length,
            torch.stack(next_state0),
            torch.stack(next_state1),
        )


def export_decoder(asr_model, filename: str):
    decoder = HaasDecoder(asr_model.decoder).eval()
    layers = asr_model.decoder.pred_rnn_layers
    width = asr_model.decoder.prediction["dec_rnn"].layout.width

    targets = torch.tensor([[asr_model.decoder.blank_idx]], dtype=torch.int32)
    target_length = torch.tensor([1], dtype=torch.int32)
    state0 = torch.zeros(layers, 1, width)
    state1 = torch.zeros(layers, 1, width)

    torch.onnx.export(
        decoder,
        (targets, target_length, state0, state1),
        filename,
        input_names=["targets", "target_length", "state0", "state1"],
        output_names=[
            "decoder_out",
            "decoder_out_length",
            "next_state0",
            "next_state1",
        ],
        dynamic_axes={
            "targets": {0: "N"},
            "target_length": {0: "N"},
            "state0": {1: "N"},
            "state1": {1: "N"},
            "decoder_out": {0: "N"},
            "decoder_out_length": {0: "N"},
            "next_state0": {1: "N"},
            "next_state1": {1: "N"},
        },
        opset_version=18,
        dynamo=False,
    )
    return width


@torch.no_grad()
def main():
    if Path("./Haas-ja-v1.0.nemo").is_file():
        asr_model = nemo_asr.models.ASRModel.restore_from(
            restore_path="./Haas-ja-v1.0.nemo", map_location="cpu"
        )
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="third-intelligence/Haas-ja-v1.0", map_location="cpu"
        )

    # The checkpoint is stored in bfloat16. Export in float32.
    asr_model = asr_model.float()
    asr_model.eval()

    with open("./tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(asr_model.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i+1}\n")
        print("Saved to tokens.txt")

    # Generate bpe.vocab for hotword support
    print("Generating bpe.vocab for hotword support...")
    generate_bpe_vocab_from_model(
        asr_model=asr_model,
        output_path="./bpe.vocab",
    )

    asr_model.encoder.export("encoder.onnx")
    decoder_state_dim = export_decoder(asr_model, "decoder.onnx")
    asr_model.joint.export("joiner.onnx")
    os.system("ls -lh *.onnx")

    normalize_type = asr_model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": asr_model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": asr_model.decoder.pred_rnn_layers,
        "pred_hidden": asr_model.decoder.pred_hidden,
        # The decoder states are not (pred_rnn_layers, N, pred_hidden)
        "decoder_state_dim": decoder_state_dim,
        "is_tdt": 1,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "2",
        "model_author": "Third Intelligence",
        "url": "https://huggingface.co/third-intelligence/Haas-ja-v1.0",
        "comment": "TDT with a Kimi Delta Attention prediction network",
        "feat_dim": 128,
    }

    for m in ["encoder", "decoder", "joiner"]:
        quantize_dynamic(
            model_input=f"./{m}.onnx",
            model_output=f"./{m}.int8.onnx",
            weight_type=QuantType.QUInt8 if m == "encoder" else QuantType.QInt8,
        )
        os.system("ls -lh *.onnx")

    add_meta_data("encoder.int8.onnx", meta_data)
    add_meta_data("encoder.onnx", meta_data)
    print("meta_data", meta_data)


if __name__ == "__main__":
    main()
