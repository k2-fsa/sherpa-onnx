#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation
#
# Export the Dolphin encoder and attention decoder to ONNX.
#
# Dolphin's released sherpa-onnx model (model.onnx) contains only the
# encoder + CTC branch, so language/region control tokens cannot be used.
# This script additionally exports:
#
#   encoder.onnx:
#       inputs
#           feats        float32 [batch=1, T, 80]   (normalized with metadata)
#           feats_len    int64   [1]
#       outputs
#           encoder_out  float32 [batch=1, T', 512]
#
#   decoder.onnx  (one autoregressive step over the full prefix, no KV cache)
#       inputs
#           encoder_out  float32 [batch=1, T', 512]
#           ys           int64   [batch=1, N]  prefix token ids, ys[0] == <sos>
#       outputs
#           logp         float32 [batch=1, vocab_size]  log-softmax for the
#                                                    next token
#
# Usage:
#   git clone https://github.com/DataoceanAI/dolphin /path/to/dolphin
#   # download model files (base.pt, train.yaml, units.txt, bpe.model,
#   # feats_stats.npz) from https://huggingface.co/DataoceanAI/dolphin-base
#   python3 ./export-onnx.py \
#       --dolphin-repo /path/to/dolphin \
#       --model-dir /path/to/dolphin-base \
#       --output-dir ./out
#
# The produced encoder.onnx and decoder.onnx pair enables the attention
# decoder path in sherpa-onnx, which honors
# --dolphin-language and --dolphin-region.
#
# The released CTC model does not expose encoder_out and cannot replace
# encoder.onnx in this pair. No CTC model is required for attention decoding.

import argparse
import math
import sys
import types
from pathlib import Path

import torch


def install_missing_dep_stubs():
    """The reference repo pulls modelscope/funasr only for downloading
    checkpoints and VAD. Stub them so the model code imports cleanly."""

    class ADict(dict):
        def __getattr__(self, k):
            return self.get(k)

        def __setattr__(self, k, v):
            self[k] = v

    for name, attr in [
        ("addict", {"Dict": ADict}),
        ("funasr", {"AutoModel": type("AutoModel", (), {})}),
        ("modelscope", {}),
        ("modelscope.models", {}),
        ("modelscope.models.audio", {}),
        ("modelscope.models.audio.funasr", {}),
        (
            "modelscope.models.audio.funasr.model",
            {"GenericFunASR": type("GenericFunASR", (), {})},
        ),
    ]:
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            mod.__path__ = []
            for k, v in attr.items():
                setattr(mod, k, v)
            sys.modules[name] = mod


def convert_v1_state_dict(state_dict):
    """Mirror of dolphin.transcribe.convert_v1_state_dict."""
    d_model = state_dict["decoder.output_layer.weight"].size(-1)
    pe = torch.zeros(5000, d_model)
    position = torch.arange(0, 5000, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    state_dict["encoder.global_cmvn.mean"] = state_dict.pop("normalize.mean")
    state_dict["encoder.global_cmvn.std"] = state_dict.pop("normalize.std")
    state_dict["decoder.embed.1.pe"] = pe
    state_dict.pop("frontend.logmel.melmat", None)
    return state_dict


def load_dolphin(model_dir: Path, dolphin_repo: Path):
    import yaml

    sys.path.insert(0, str(dolphin_repo))
    install_missing_dep_stubs()

    from dolphin.model import init_speech_model

    configs = yaml.safe_load(open(model_dir / "train.yaml", encoding="utf-8"))
    configs.setdefault("cmvn_conf", {})["cmvn_file"] = str(
        model_dir / "feats_stats.npz"
    )
    configs.setdefault("tokenizer_conf", {})
    configs["tokenizer_conf"]["symbol_table_path"] = str(model_dir / "units.txt")
    if (model_dir / "bpe.model").exists():
        configs["tokenizer_conf"]["bpe_path"] = str(model_dir / "bpe.model")

    model = init_speech_model(configs)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.eps = float(configs.get("layer_norm_eps", module.eps))

    ckpt = next(iter(sorted(model_dir.glob("*.pt"))))
    state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
    state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("context_module.")
    }
    if "normalize.mean" in state_dict:
        state_dict = convert_v1_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not missing, f"missing keys: {missing[:10]}"
    assert not unexpected, f"unexpected keys: {unexpected[:10]}"
    model.eval()
    return model


class EncoderWrapper(torch.nn.Module):
    """feats [1,T,80] + feats_len [1] -> encoder_out [1,T',D].

    Follows the convention of the released CTC model.onnx: global_cmvn is
    stripped from the graph and mean/invstd are stored as model metadata;
    sherpa-onnx normalizes the features in C++ before the forward.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats, feats_len):
        encoder_out, encoder_mask = self.encoder(feats, feats_len)
        return encoder_out


class DecoderWrapper(torch.nn.Module):
    """encoder_out [1,T',D] + ys [1,N] -> logp [1,V] for the next token.

    Mirrors TransformerDecoder.forward_one_step with an empty cache:
    embed the whole prefix, run the decoder blocks with a causal mask, and
    return log-softmax of the last position.
    """

    def __init__(self, decoder):
        super().__init__()
        assert not decoder.use_sdpa, "export with use_sdpa=False"
        self.decoder = decoder

    def forward(self, encoder_out, ys):
        from dolphin.mask import subsequent_mask

        batch_size, t_enc, _ = encoder_out.shape
        n = ys.shape[1]

        memory_mask = torch.ones(
            batch_size, 1, t_enc, dtype=torch.bool, device=ys.device
        )
        tgt_mask = subsequent_mask(n, device=ys.device).unsqueeze(0)

        x, _ = self.decoder.embed(ys)
        for layer in self.decoder.decoders:
            x, tgt_mask, _, memory_mask = layer(
                x, tgt_mask, encoder_out, memory_mask, cache=None
            )

        if self.decoder.normalize_before:
            x = self.decoder.after_norm(x)

        y = self.decoder.output_layer(x[:, -1])
        return torch.log_softmax(y, dim=-1)


def export(model, model_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strip global_cmvn: sherpa-onnx normalizes in C++ using the mean/invstd
    # stored in model metadata, same as the released CTC model.onnx.
    model.encoder.global_cmvn = None

    feats = torch.randn(1, 300, 80)
    feats_len = torch.tensor([300])
    with torch.no_grad():
        enc_out = EncoderWrapper(model.encoder)(feats, feats_len)
    print(f"encoder_out shape: {enc_out.shape}")

    torch.onnx.export(
        EncoderWrapper(model.encoder),
        (feats, feats_len),
        str(output_dir / "encoder.onnx"),
        input_names=["feats", "feats_len"],
        output_names=["encoder_out"],
        dynamic_axes={
            "feats": {1: "T"},
            "encoder_out": {1: "T_out"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("exported encoder.onnx")

    ys = torch.tensor([[model.sos]], dtype=torch.long)
    dec = DecoderWrapper(model.decoder)
    with torch.no_grad():
        logp = dec(enc_out, ys)
    print(f"decoder logp shape: {logp.shape}")

    torch.onnx.export(
        dec,
        (enc_out, ys),
        str(output_dir / "decoder.onnx"),
        input_names=["encoder_out", "ys"],
        output_names=["logp"],
        dynamic_axes={
            "encoder_out": {1: "T_out"},
            "ys": {1: "N"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("exported decoder.onnx")

    meta = {
        "vocab_size": str(model.vocab_size),
        "sos": str(model.sos),
        "eos": str(model.eos),
        "model_type": "dolphin",
        "version": "1",
        "model_author": "DataoceanAI",
        "comment": "attention decoder for language/region control; "
        "input ys is the full token prefix starting with <sos>; output "
        "logp is log-softmax of the next-token distribution",
    }

    # mean/invstd of the stripped global_cmvn, same convention as the
    # released CTC model.onnx (metadata keys "mean" and "invstd").
    stats = __import__("numpy").load(str(model_dir / "feats_stats.npz"))
    count = stats["count"]
    mean = stats["sum"] / count
    var = stats["sum_square"] / count - mean * mean
    invstd = 1.0 / __import__("numpy").sqrt(__import__("numpy").maximum(var, 1.0e-20))
    meta["mean"] = ",".join(str(float(v)) for v in mean)
    meta["invstd"] = ",".join(str(float(v)) for v in invstd)

    import onnx

    for name in ("encoder.onnx", "decoder.onnx"):
        p = output_dir / name
        m = onnx.load(str(p))
        del m.metadata_props[:]
        for k, v in meta.items():
            e = m.metadata_props.add()
            e.key = k
            e.value = v
        onnx.save(m, str(p))

    # Dolphin units already use the symbol/id format expected by sherpa-onnx.
    (output_dir / "tokens.txt").write_text(
        (model_dir / "units.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )

    print("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dolphin-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = load_dolphin(args.model_dir, args.dolphin_repo)
    export(model, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
