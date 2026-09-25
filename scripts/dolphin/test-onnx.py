#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation
#
# Verify exported encoder.onnx / decoder.onnx against the PyTorch
# reference implementation, and simulate the C++ greedy decode loop
# (forced lang/region prefix + header tokens + text until <eos>).

import argparse
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torchaudio


def install_missing_dep_stubs():
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


def load_dolphin(model_dir: Path, dolphin_repo: Path):
    import yaml

    sys.path.insert(0, str(dolphin_repo))
    install_missing_dep_stubs()
    from dolphin.model import init_speech_model

    configs = yaml.safe_load(open(model_dir / "train.yaml", encoding="utf-8"))
    configs["cmvn_conf"]["cmvn_file"] = str(model_dir / "feats_stats.npz")
    configs.setdefault("tokenizer_conf", {})
    configs["tokenizer_conf"]["symbol_table_path"] = str(model_dir / "units.txt")
    if (model_dir / "bpe.model").exists():
        configs["tokenizer_conf"]["bpe_path"] = str(model_dir / "bpe.model")
    model = init_speech_model(configs)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.eps = float(configs.get("layer_norm_eps", module.eps))

    ckpt = next(model_dir.glob("*.pt"))
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("context_module.")}
    if "normalize.mean" in sd:
        d_model = sd["decoder.output_layer.weight"].size(-1)
        pe = torch.zeros(5000, d_model)
        pos = torch.arange(0, 5000, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        sd["encoder.global_cmvn.mean"] = sd.pop("normalize.mean")
        sd["encoder.global_cmvn.std"] = sd.pop("normalize.std")
        sd["decoder.embed.1.pe"] = pe.unsqueeze(0)
        sd.pop("frontend.logmel.melmat", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected
    model.eval()
    return model


def load_id2token(units):
    id2token = {}
    token2id = {}
    for line in open(units, encoding="utf-8"):
        parts = line.split()
        if len(parts) == 2:
            token2id[parts[0]] = int(parts[1])
            id2token[int(parts[1])] = parts[0]
    return id2token, token2id


def get_feats(wav_path, model_dir):
    """Match the reference pipeline: DefaultFrontend (STFT + power + LogMel),
    as in dolphin.processor.extract_feats."""
    import soundfile as sf
    import yaml

    from dolphin.model import DefaultFrontend

    data, sr = sf.read(str(wav_path), dtype="float32")
    wav = torch.from_numpy(data)
    if wav.dim() == 2:
        wav = wav[:, 0]
    wav = wav.unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    configs = yaml.safe_load(open(model_dir / "train.yaml", encoding="utf-8"))
    fe_conf = dict(configs["dataset_conf"]["frontend_conf"])
    if isinstance(fe_conf.get("fs"), str):
        fe_conf["fs"] = int(fe_conf["fs"].replace("k", "000"))
    frontend = DefaultFrontend(**fe_conf)
    frontend.eval()
    lens = torch.tensor([wav.size(-1)], dtype=torch.long)
    with torch.no_grad():
        mel, feature_lengths = frontend(wav, lens)
    return mel, feature_lengths


@torch.no_grad()
def torch_decode(
    model,
    enc_out,
    enc_mask,
    sos,
    eos,
    no_tm,
    lang_id=None,
    region_id=None,
    max_len=None,
):
    """Greedy decode mirroring predict_lang_region_timestamp + text loop."""
    from dolphin.mask import subsequent_mask

    ys = [sos]
    if lang_id is not None:
        ys.append(lang_id)
        if region_id is not None:
            ys.append(region_id)
    if max_len is None:
        max_len = enc_out.shape[1]

    def step(prefix):
        ys_t = torch.tensor([prefix], dtype=torch.long)
        mask = subsequent_mask(len(prefix)).unsqueeze(0)
        logp = model.decoder.forward_one_step(
            enc_out,
            enc_mask,
            ys_t,
            mask,
            {"self_att_cache": {}, "cross_att_cache": {}},
        )
        return logp[0]

    # header tokens: lang, region, <asr>, <notimestamp>
    while len(ys) < 5:
        if len(ys) == 4:
            ys.append(no_tm)
        else:
            ys.append(int(step(ys).argmax()))

    while len(ys) <= max_len:
        nxt = int(step(ys).argmax())
        ys.append(nxt)
        if nxt == eos:
            break
    return ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolphin-repo", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--lang", type=str, default=None)
    ap.add_argument("--region", type=str, default=None)
    ap.add_argument("--sherpa-onnx-offline", type=Path, required=True)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()
    if args.region and not args.lang:
        ap.error("--region requires --lang")

    model = load_dolphin(args.model_dir, args.dolphin_repo)
    id2token, token2id = load_id2token(args.model_dir / "units.txt")

    feats, feats_len = get_feats(args.wav, args.model_dir)

    # ---- PyTorch reference ----
    # NOTE: GlobalMVN normalizes in-place, so clone before the encoder call.
    feats_raw = feats.clone()
    with torch.no_grad():
        enc_out, enc_mask = model.encoder(feats_raw, feats_len)
    print("encoder_out:", tuple(enc_out.shape))

    lang_id = token2id[f"<{args.lang}>"] if args.lang else None
    region_id = token2id[f"<{args.region}>"] if args.region else None

    no_tm = token2id["<notimestamp>"]
    pt_ys = torch_decode(
        model,
        enc_out,
        enc_mask,
        model.sos,
        model.eos,
        no_tm,
        lang_id=lang_id,
        region_id=region_id,
    )
    pt_tokens = pt_ys[5:]
    if pt_tokens and pt_tokens[-1] == model.eos:
        pt_tokens = pt_tokens[:-1]
    print("torch header:", [id2token.get(t, t) for t in pt_ys[1:5]])
    print("torch text tokens:", [id2token.get(t, t) for t in pt_tokens])

    # ---- ONNX ----
    import onnxruntime as ort

    enc_sess = ort.InferenceSession(
        str(args.onnx_dir / "encoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    dec_sess = ort.InferenceSession(
        str(args.onnx_dir / "decoder.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # encoder parity (cmvn stripped -> normalize like sherpa-onnx does)
    stats = np.load(args.model_dir / "feats_stats.npz")
    count = stats["count"]
    mean = stats["sum"] / count
    var = stats["sum_square"] / count - mean * mean
    invstd = 1.0 / np.sqrt(np.maximum(var, 1e-20))
    feats_norm = (feats.numpy() - mean) * invstd  # feats is unmodified raw

    enc_out_onnx = enc_sess.run(
        ["encoder_out"],
        {
            "feats": feats_norm.astype(np.float32),
            "feats_len": feats_len.numpy(),
        },
    )[0]
    diff = np.abs(enc_out_onnx - enc_out.numpy()).max()
    print(f"encoder_out max diff: {diff:.6f}")
    np.testing.assert_allclose(
        enc_out_onnx, enc_out.numpy(), rtol=args.rtol, atol=args.atol
    )

    def onnx_step(ys):
        logp = dec_sess.run(
            ["logp"],
            {
                "encoder_out": enc_out_onnx,
                "ys": np.array([ys], dtype=np.int64),
            },
        )[0][0]
        return logp

    ys = [model.sos]
    if lang_id is not None:
        ys.append(lang_id)
        if region_id is not None:
            ys.append(region_id)

    # header: positions len(ys)..4 -> lang, region, task, timestamp
    while len(ys) < 5:
        if len(ys) == 4:
            ys.append(no_tm)
        else:
            logp = onnx_step(ys)
            ys.append(int(logp.argmax()))

    print("onnx header:", [id2token.get(t, t) for t in ys[1:]])

    # text loop
    while len(ys) <= enc_out_onnx.shape[1]:
        logp = onnx_step(ys)
        nxt = int(logp.argmax())
        ys.append(nxt)
        if nxt == model.eos:
            break

    tokens = ys[5:]
    if tokens and tokens[-1] == model.eos:
        tokens = tokens[:-1]
    print("onnx text tokens:", [id2token.get(t, t) for t in tokens])
    print("lang:", id2token.get(ys[1]), "region:", id2token.get(ys[2]))

    # Include the header, sequence length and EOS in the comparison.
    np.testing.assert_array_equal(ys, pt_ys)

    command = [
        str(args.sherpa_onnx_offline.resolve()),
        f"--dolphin-encoder={args.onnx_dir / 'encoder.onnx'}",
        f"--dolphin-decoder={args.onnx_dir / 'decoder.onnx'}",
        f"--tokens={args.onnx_dir / 'tokens.txt'}",
        f"--dolphin-language={args.lang or ''}",
        f"--dolphin-region={args.region or ''}",
        str(args.wav),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    results = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    if len(results) != 1:
        raise AssertionError(f"Expected one native result: {completed.stdout}")
    result = results[0]
    expected_tokens = [id2token[t].replace("▁", " ") for t in tokens if t != model.sos]
    if result["tokens"] != expected_tokens:
        raise AssertionError((result["tokens"], expected_tokens))
    if result["lang"] != id2token[ys[1]][1:-1]:
        raise AssertionError((result["lang"], id2token[ys[1]]))
    print("PyTorch, ONNX and native recognizer parity passed")


if __name__ == "__main__":
    main()
