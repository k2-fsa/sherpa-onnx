#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/export_torch.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Export one vosk model to a PyTorch model directory — the middle stage of
# the pipeline:
#
#     vosk model dir  --export_torch.py-->  torch model dir
#     --export_onnx.py-->  onnx model dir   --compare_onnx.py-->  verified
#
# The output keeps the ORIGINAL model directory layout: everything is
# copied through except am/final.mdl, which is replaced by the pieces
# below.  final.mdl is never written to the output.
#
#   <out_dir>/
#   ├── am/
#   │   ├── am.pt                 torch model: {"state_dict", "args"}
#   │   ├── am_flat.py            auto-generated explicit-wiring nn.Module
#   │   │                         (readable/editable form of the same net;
#   │   │                         parity-checked against am.pt, see
#   │   │                         flatten.py and README "Readable model code")
#   │   ├── model_meta.json       dims / subsampling / context
#   │   └── transition_model.mdl  the TransitionModel half of final.mdl
#   ├── conf/  ivector/  graph/   copied unchanged
#   └── README                    copied unchanged
#
# am.pt format: a dict with
#   * "state_dict": the module's state_dict
#   * "args":       everything needed to rebuild the module from scratch
#                   (the parsed network as plain python objects;
#                   Nnet3Torch.from_export_args(args) reconstructs it)
# so export_onnx.py can load the dict, reconstruct, load weights and export.
#
# The export refuses to save anything that does not pass a sanity forward
# (correct output dim, finite values), and the model can be re-verified at
# any time with compare_torch.py --model-dir <in_dir>.
#
# Usage:
#   python3 export_torch.py --in-dir ./vosk-model-small-cn-0.22 \
#       --out-dir ./torch-model-cn

import argparse
import json
import shutil
from pathlib import Path

from recognizer import default_kaldi_src, read_model_conf

import numpy as np
import torch


from nnet3.mdl_reader import dump_via_kaldi, split_transition_model_bytes
from torch_model import Nnet3Torch
from flatten import build_flat, check_parity, generate_flat_module, remap_state_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="vosk model dir (am/final.mdl, conf/, ivector/, graph/)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="output model dir: original layout with am/am.pt",
    )
    parser.add_argument(
        "--kaldi-src",
        type=str,
        default=None,
        help="kaldi src dir providing nnet3bin/nnet3-am-copy "
        "(default: $KALDI_ROOT/src, /opt/kaldi/src, or ../kaldi/src)",
    )
    parser.add_argument(
        "--nnet3-am-copy",
        type=str,
        default=None,
        help="explicit path of nnet3-am-copy (overrides --kaldi-src)",
    )
    return parser.parse_args()


def copy_model_tree(in_dir: Path, out_dir: Path) -> None:
    """Copy the model dir except am/final.mdl: the output keeps the
    original layout but replaces the AM; final.mdl is not shipped)."""
    for src in sorted(in_dir.rglob("*")):
        rel = src.relative_to(in_dir)
        if rel == Path("am/final.mdl"):
            continue
        dst = out_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main():
    args = get_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    nnet3_am_copy = args.nnet3_am_copy or (
        f"{args.kaldi_src or default_kaldi_src()}/nnet3bin/nnet3-am-copy"
    )

    conf = read_model_conf(in_dir / "conf" / "model.conf")
    factor = int(conf["frame-subsampling-factor"])  # required, never assumed
    acoustic_scale = float(conf.get("acoustic-scale", "1.0"))
    print(
        f"model.conf: frame_subsampling_factor={factor} "
        f"acoustic_scale={acoustic_scale}"
    )

    # ---- 1. parse final.mdl (streamed text; see nnet3/notes.md) --------
    print(f"parsing {in_dir / 'am' / 'final.mdl'} via {nnet3_am_copy} ...")
    spec = dump_via_kaldi(str(in_dir / "am" / "final.mdl"), nnet3_am_copy=nnet3_am_copy)
    print(
        f"network: input={spec.input_dim} ivector={spec.ivector_dim} "
        f"output={spec.output_dim}  components={len(spec.components)} "
        f"L/R={spec.left_context}/{spec.right_context}"
    )

    # ---- 2. build the torch model ------------------------------------
    net = Nnet3Torch(spec)
    net.eval()

    # ---- 3. sanity forward before saving -----------------------------
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((1, 40, spec.input_dim)).astype(np.float32)
    iv = (rng.standard_normal((1, spec.ivector_dim)) * 0.1).astype(np.float32)
    with torch.no_grad():
        out = net(torch.from_numpy(feats), torch.from_numpy(iv))
    assert out.shape == (1, 40, spec.output_dim), out.shape
    assert torch.isfinite(out).all(), "non-finite sanity output"
    print(f"sanity forward OK: {tuple(out.shape)}")

    # ---- 3b. flattened explicit-wiring module (am_flat.py) -------------
    # Same math as Nnet3Torch, but the node graph is spelled out as plain
    # code (readable / editable).  Verified against Nnet3Torch here, so a
    # generated file that drifts never gets written silently.
    flat_code = generate_flat_module(net.spec, source=in_dir.name)
    flat = build_flat(net.spec, source=in_dir.name)
    flat.load_state_dict(remap_state_dict(net.state_dict(), net.spec))
    worst = 0.0
    for t_len in (1, 7, 40, 137):
        feats = torch.randn(2, t_len, spec.input_dim)
        iv = 0.1 * torch.randn(2, spec.ivector_dim)
        worst = max(worst, check_parity(net, flat, feats, iv))
    assert worst < 1e-6, f"flat module parity failed: {worst:.2e}"
    print(f"flat module parity OK (max diff {worst:.2e})")

    # ---- 4. write the model tree -------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_model_tree(in_dir, out_dir)

    am_dir = out_dir / "am"
    payload = {"state_dict": net.state_dict(), "args": net.export_args()}
    torch.save(payload, am_dir / "am.pt")
    (am_dir / "am_flat.py").write_text(flat_code)

    (am_dir / "transition_model.mdl").write_bytes(
        split_transition_model_bytes(str(in_dir / "am" / "final.mdl"))
    )

    mfcc_conf = read_model_conf(in_dir / "conf" / "mfcc.conf")
    meta = {
        "source_model": in_dir.name,
        "sample_rate": int(float(mfcc_conf.get("sample-frequency", 16000))),
        "input_dim": spec.input_dim,
        "ivector_dim": spec.ivector_dim,
        "num_pdfs": spec.output_dim,
        # decoder row k = network output at input frame k*factor (kaldi's
        # request steps t += frame_subsampling_factor, no offset)
        "frame_subsampling_factor": factor,
        "acoustic_scale": acoustic_scale,
        "left_context": spec.left_context,
        "right_context": spec.right_context,
        "components": {
            c.type: sum(1 for x in spec.components if x.type == c.type)
            for c in spec.components
        },
    }
    (am_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {out_dir}")
    print("  am/am.pt                (state_dict + args)")
    print("  am/am_flat.py           (explicit-wiring module, parity-checked)")
    print("  am/model_meta.json")
    print("  am/transition_model.mdl")
    print("verify any time with: compare_torch.py --model-dir", in_dir)


if __name__ == "__main__":
    main()
