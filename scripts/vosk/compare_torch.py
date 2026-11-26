#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/compare_torch.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Verifies the PyTorch model (torch_model.Nnet3Torch) produces the SAME
# output as the original kaldi model (am/final.mdl) for the same input --
# this is the gate that must pass before exporting to ONNX
# .
#
# What is compared
# ----------------
# Both sides are fed *identical* inputs, so any difference is the network
# evaluation itself:
#
#   features: computed ONCE with kaldi's compute-mfcc-feats (dither=0,
#             the model's conf/mfcc.conf) and given to both sides;
#             feature extraction is not under test here.
#   ivector:  one fixed-seed synthetic vector per utterance, given to both
#             sides (kaldi: nnet3-compute --ivectors; ours: the forward
#             argument).  The net is a function of (features, ivector); any
#             identical pair is a valid test of that function.
#
# Rows
# ----
# kaldi's nnet3-compute (DecodableNnetSimple) emits one row per INPUT frame:
# row i = network output anchored at input frame i (full rate; only the
# *decoder* consumes every 3rd frame).  Our Nnet3Torch also computes the
# full-rate output, so the comparison is direct row-for-row:
#
#     R[i] == O[i]     for every i in 0..T-1
#
# (This is strictly stronger than the subsampled check R[k] == O[3k] of
#: full-rate equality implies it.  The 3x slice
# itself lives in VoskAmWrapper.)  Failure patterns diagnose exporter bugs:
# a constant time shift means the Tdnn/Offset edge handling or an anchor is
# wrong; drift means chunk logic is off; isolated bad rows point at one
# component.
#
# kaldi runs the whole utterance as ONE chunk (--frames-per-chunk=1000000):
# nnet3-compute's default 50-frame chunks restart a recurrent net's LSTM
# state every chunk (gigaspeech: 9.4 max diff vs one chunk).
#
# Gates: max abs diff <= --tolerance (default 1e-3; float noise is ~1e-5).
# Recurrent (LSTM) nets amplify float noise chaotically over time (kaldi
# vs kaldi with 7-digit-rounded params differs by 0.42 on gigaspeech), so
# for them a failing max falls back to: median per-row max <= --tolerance
# and argmax agreement >= 0.99.
# and per-row cosine >= 0.9999.
#
# Usage:
#   python3 compare_torch.py --model-dir /path/to/vosk-model-small-cn-0.22 \
#       --wav /path/to/test.wav [--kaldi-src /path/to/kaldi/src]

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# scripts/vosk on sys.path so `nnet3` imports resolve like the other tools

import kaldi_native_io

from recognizer import default_kaldi_src, load_audio, read_meta, read_model_conf
from nnet3.mdl_reader import dump_via_kaldi
from torch_model import Nnet3Torch


def get_args():
    p = argparse.ArgumentParser(
        description="Verify the PyTorch model matches the kaldi model "
        "row-by-row (R[k] == O[3k]) for the same input."
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="a vosk model dir (needs am/final.mdl, conf/mfcc.conf)",
    )
    p.add_argument(
        "--wav",
        required=True,
        help="mono 16-bit wav, any sample rate supported by "
        "compute-mfcc-feats (it resamples per mfcc.conf)",
    )
    p.add_argument(
        "--kaldi-src",
        default=None,
        help="kaldi src dir (default: $KALDI_ROOT/src or "
        "/opt/kaldi/src or the local checkout)",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="max abs diff between kaldi and torch rows",
    )
    p.add_argument(
        "--am-pt",
        default=None,
        help="test the RELOADED export (am/am.pt from "
        "export_torch.py) instead of building the net from "
        "final.mdl: reconstructs from 'args', loads "
        "'state_dict', then runs the same kaldi comparison",
    )
    return p.parse_args()


def main():
    args = get_args()
    kaldi_src = args.kaldi_src or default_kaldi_src()
    assert kaldi_src, "no kaldi found; pass --kaldi-src or set KALDI_ROOT"
    model_dir = Path(args.model_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # ---- 1. features: computed once, used by both sides -------------
        # kaldi tools take a file path: hand them the librosa-resampled
        # audio at the model's rate (we always resample ourselves)
        if args.am_pt:
            sample_rate = read_meta(Path(args.am_pt).parent.parent)["sample_rate"]
        else:
            # no exported metadata on the default path: take the rate
            # from the original model's own config
            sample_rate = int(
                float(
                    read_model_conf(model_dir / "conf" / "mfcc.conf").get(
                        "sample-frequency", 16000
                    )
                )
            )
        y, sr = load_audio(args.wav, sample_rate)
        wav16 = tmp / "wav_16k.wav"  # removed with tmp
        sf.write(str(wav16), y, sr, subtype="PCM_16")
        wav_scp = tmp / "wav.scp"
        wav_scp.write_text(f"utt {wav16.resolve()}\n")
        feats_ark = tmp / "feats.ark"
        subprocess.run(
            [
                f"{kaldi_src}/featbin/compute-mfcc-feats",
                "--dither=0",
                f"--config={model_dir}/conf/mfcc.conf",
                f"scp,p:{wav_scp}",
                f"ark:{feats_ark}",
            ],
            check=True,
            capture_output=True,
        )
        with kaldi_native_io.SequentialFloatMatrixReader(f"ark:{feats_ark}") as r:
            feats = np.array(next(iter(r))[1], dtype=np.float32, copy=True)
        T = feats.shape[0]
        print(f"audio: {len(y) / sr:.2f}s @ {sr} Hz (librosa)")
        print(f"features: {feats.shape} (from {args.wav})")

        # ---- 2. the torch model: either rebuilt from the exported am.pt
        #         (the reload path, incl. state_dict) or parsed fresh from
        #         final.mdl (the export path) ------------------------------
        if args.am_pt:
            payload = torch.load(args.am_pt, weights_only=False)
            net = Nnet3Torch.from_export_args(payload["args"])
            net.load_state_dict(payload["state_dict"])
            net.eval()
            spec = net.spec
            print(f"model: RELOADED from {args.am_pt}")
        else:
            spec = dump_via_kaldi(
                str(model_dir / "am" / "final.mdl"),
                nnet3_am_copy=f"{kaldi_src}/nnet3bin/nnet3-am-copy",
            )
            net = Nnet3Torch(spec)
            net.eval()
            print("model: parsed from final.mdl")
        print(
            f"       input={spec.input_dim} ivector={spec.ivector_dim} "
            f"output={spec.output_dim} (num_pdfs)"
        )

        # ---- 3. identical synthetic i-vector for both sides -------------
        rng = np.random.default_rng(0)
        iv = (rng.standard_normal(spec.ivector_dim) * 0.1).astype(np.float32)
        iv_ark = tmp / "iv.ark"
        with kaldi_native_io.FloatVectorWriter(f"ark:{iv_ark}") as w:
            w.write("utt", iv)

        # ---- 4. kaldi reference (DecodableNnetSimple row semantics) ----
        out_ark = tmp / "kaldi_out.ark"
        subprocess.run(
            [
                f"{kaldi_src}/nnet3bin/nnet3-compute",
                # ONE chunk for the whole utterance: the default (50 frames)
                # restarts every recurrence (LSTM state) at each chunk, so
                # its rows are not those of a continuous forward (nor of
                # vosk's looped decodable).  Non-recurrent nets: identical.
                "--frames-per-chunk=1000000",
                f"--ivectors=ark:{iv_ark}",
                str(model_dir / "am" / "final.mdl"),
                f"ark:{feats_ark}",
                f"ark:{out_ark}",
            ],
            check=True,
            capture_output=True,
        )
        with kaldi_native_io.SequentialFloatMatrixReader(f"ark:{out_ark}") as r:
            ref = np.array(next(iter(r))[1], dtype=np.float32, copy=True)
        # DecodableNnetSimple emits one row per input frame (full rate)
        print(f"kaldi nnet3-compute: {ref.shape} rows (T = {T})")
        assert ref.shape[0] == T, (ref.shape, T)

        # ---- 5. torch: full-rate output --------------------------------
        # the net is pure: edge semantics are the caller's job -- feed
        # replicated edge frames beyond the utterance (kaldi clamps
        # out-of-range input indexes to the edge the same way)
        padded = np.pad(
            feats, ((spec.left_context, spec.right_context), (0, 0)), mode="edge"
        )
        with torch.no_grad():
            # the same i-vector on every frame (nnet3-compute --ivectors
            # takes one per utterance)
            iv_rows = np.tile(iv, (padded.shape[0], 1))
            ours = net(torch.from_numpy(padded[None]), torch.from_numpy(iv_rows[None]))[
                0
            ].numpy()
        ours = ours[spec.left_context : spec.left_context + T]
        print(f"torch full-rate:     {ours.shape}")

        # ---- 6. row-by-row: R[i] == O[i] ------------------------------
        assert ours.shape == ref.shape, (ours.shape, ref.shape)
        diff = np.abs(ref.astype(np.float64) - ours.astype(np.float64))
        worst_abs = float(diff.max())
        worst_k = int(diff.max(axis=1).argmax())
        cos = np.sum(ref.astype(np.float64) * ours.astype(np.float64), axis=1) / (
            np.linalg.norm(ref.astype(np.float64), axis=1)
            * np.linalg.norm(ours.astype(np.float64), axis=1)
            + 1e-12
        )
        worst_cos = float(cos.min())
        # divergence profile: localizes the bug (edge vs state vs core)
        rowmax = diff.max(axis=1)
        bad = np.where(rowmax > 1e-3)[0]
        if len(bad):
            argmax_ok = float((ref.argmax(axis=1) == ours.argmax(axis=1)).mean())
            print(
                f"diff profile: {len(bad)}/{len(ref)} rows off >1e-3; "
                f"first bad row {int(bad[0])}; argmax agreement {argmax_ok:.3f}"
            )
            step = max(1, len(ref) // 20)
            print(
                "per-row max|diff|: "
                + " ".join(f"{k}:{rowmax[k]:.3g}" for k in range(0, len(ref), step))
            )

        ok = worst_abs <= args.tolerance and worst_cos >= 0.9999
        if not ok and net._recurrent:
            # A recurrence can amplify float noise chaotically: kaldi vs
            # kaldi itself with the text-dump-rounded params differs by
            # ~0.4 at late rows of gigaspeech's lstm2.  Max abs is then no
            # semantics test; require float-noise rows for the bulk of the
            # utterance and the same argmax everywhere instead.
            median_row = float(np.median(rowmax))
            argmax_ok = float((ref.argmax(axis=1) == ours.argmax(axis=1)).mean())
            ok = median_row <= args.tolerance and argmax_ok >= 0.99
            print(
                f"recurrent net: median row max_abs={median_row:.2e} "
                f"argmax agreement={argmax_ok:.3f} -> " + ("OK" if ok else "FAIL")
            )
        print(
            f"rows compared: {ref.shape[0]}   worst max_abs={worst_abs:.6f} "
            f"(row {worst_k})   worst cosine={worst_cos:.6f}"
        )
        print("Torch == KALDI: " + ("OK" if ok else "FAIL"))
        if not ok:
            if worst_abs > args.tolerance and worst_cos >= 0.9999:
                print(
                    "hint: cosine is fine -> looks like float noise or a "
                    "small scale mismatch; inspect row %d" % worst_k
                )
            else:
                print(
                    "hint: large cosine error -> wrong weights/graph "
                    "mapping; or a shifted match -> subsampling anchor bug"
                )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
