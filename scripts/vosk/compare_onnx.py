#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/compare_onnx.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Tests an exported ONNX vosk model (export_onnx.py output) — the validation
# gate of, in two stages:
#
# STAGE 1 — AM level: the ONNX model vs the original kaldi model
#   * identical features (kaldi compute-mfcc-feats, dither=0) and an
#     identical fixed-seed i-vector are fed to both:
#       kaldi: nnet3-compute am/final.mdl   -> R[i]   (one row per input frame)
#       ours : onnxruntime  am/final.onnx   -> O[k]   (rows 0::3 kept, so
#              O[k] must equal R[3*k] -- row k is anchored at input
#              frame 3*k, the convention kaldi's decodables use)
#   * gate: max abs diff <= 1e-3, per-row cosine >= 0.9999
#
# STAGE 2 — decode level: wav -> text through the full Python stack
#   (onnxruntime + kaldi-decoder + kaldifst + kaldi-native-fbank +
#   kaldi-hmm-gmm + kaldi-native-io):
#
#   wav (soundfile, python-api-examples convention)
#     -> kaldi_native_fbank.OnlineMfcc(conf/mfcc.conf)     features
#     -> kaldi_hmm_gmm.OnlineIvectorExtractor(ivector/)    i-vector
#     -> onnxruntime (am/final.onnx)                       log_probs
#     -> kaldi_decoder C++ matrix decodable over log_probs  (trans-id scores)
#     -> kaldifst graph (HCLG, or HCLr+Gr composed) + kaldi_hmm_gmm.TransitionModel
#     -> decode_utterance_lattice_faster + lattice_best_path -> word ids
#     -> words.txt -> text
#   gate: text equals --expect, ignoring whitespace.  The vosk package's
#   result (same model and wav) is printed for reference only.
#
# Usage:
#   python3 compare_onnx.py --model-dir ./onnx-model-cn \
#       --ref-model-dir ./vosk-model-small-cn-0.22 --wav ./zh.wav \
#       [--kaldi-src /path/to/kaldi/src] [--skip-stage1]

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


import kaldi_native_io

from recognizer import (
    INT16_SCALE,
    OfflineRecognizer,
    OnnxAm,
    default_kaldi_src,
    load_audio,
    read_meta,
    round_up,
    run,
)


def norm(text, drop_space=False):
    """Normalize for comparison; drop_space=True ignores whitespace
    (CJK word spacing is cosmetic)."""
    t = re.sub(r"\s+", " ", text.strip())
    return t.replace(" ", "") if drop_space else t


def stage1_am_parity(
    model_dir: Path, ref_dir: Path, wav: Path, kaldi_src: str, tolerance: float
) -> None:
    """ONNX (exported dir) vs kaldi (original model's final.mdl).

    The exported dir has no final.mdl by design, so the reference model is
    passed separately (--ref-model-dir).  Features come from the original
    model's conf (identical in both dirs -- it is copied through).
    """
    print("== stage 1: ONNX vs kaldi nnet3-compute ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # identical features for both sides
        # kaldi tools take a file path: hand them the librosa-resampled
        # 16 kHz audio (we always resample ourselves)
        y, sr = load_audio(wav, read_meta(model_dir)["sample_rate"])
        wav16 = tmp / "wav_16k.wav"  # removed with tmp
        sf.write(str(wav16), y, sr, subtype="PCM_16")
        (tmp / "w.scp").write_text(f"u {wav16.resolve()}\n")
        feats_ark = tmp / "feats.ark"
        run(
            [
                f"{kaldi_src}/featbin/compute-mfcc-feats",
                "--dither=0",
                f"--config={ref_dir}/conf/mfcc.conf",
                f"scp,p:{tmp / 'w.scp'}",
                f"ark:{feats_ark}",
            ]
        )
        with kaldi_native_io.SequentialFloatMatrixReader(f"ark:{feats_ark}") as r:
            feats = np.array(next(iter(r))[1], dtype=np.float32, copy=True)
        T = feats.shape[0]

        # identical synthetic i-vector (the net is a function of (feats, iv))
        # dims from the onnx session
        sess = ort.InferenceSession(
            str(model_dir / "am" / "final.onnx"), providers=["CPUExecutionProvider"]
        )
        in_dims = {i.name: i.shape for i in sess.get_inputs()}
        iv_dim = in_dims["ivector"][1]
        rng = np.random.default_rng(0)
        iv = (rng.standard_normal(iv_dim) * 0.1).astype(np.float32)
        iv_ark = tmp / "iv.ark"
        with kaldi_native_io.FloatVectorWriter(f"ark:{iv_ark}") as w:
            w.write("u", iv)

        # kaldi reference: one row per input frame
        out_ark = tmp / "kaldi.ark"
        run(
            [
                f"{kaldi_src}/nnet3bin/nnet3-compute",
                f"--ivectors=ark:{iv_ark}",
                str(ref_dir / "am" / "final.mdl"),
                f"ark:{feats_ark}",
                f"ark:{out_ark}",
            ]
        )
        with kaldi_native_io.SequentialFloatMatrixReader(f"ark:{out_ark}") as r:
            ref = np.array(next(iter(r))[1], dtype=np.float32, copy=True)

        # ours: one row per `factor` input frames (row k anchored at
        # input frame k*factor).  The net is pure: feed replicated edge
        # frames beyond the utterance (kaldi's edge rule), on the row
        # grid so the output rows line up with the anchors.
        meta = read_meta(model_dir)
        factor = int(meta["frame_subsampling_factor"])
        pad_left = round_up(meta["left_context"], factor)
        padded = np.pad(
            feats, ((pad_left, meta["right_context"]), (0, 0)), mode="edge"
        )
        got = sess.run(None, {"features": padded[None], "ivector": iv[None]})[0][0]
        first_row = pad_left // factor
        got = got[first_row : first_row + (T + factor - 1) // factor]
        assert got.shape[0] == (T + factor - 1) // factor, (got.shape, T)

        ref_rows = ref[::factor].astype(np.float64)[: got.shape[0]]
        got_rows = got.astype(np.float64)
        worst_abs = float(np.abs(ref_rows - got_rows).max())
        cos = np.sum(ref_rows * got_rows, axis=1) / (
            np.linalg.norm(ref_rows, axis=1)
            * np.linalg.norm(got_rows, axis=1)
            + 1e-12
        )
        ok = worst_abs <= tolerance and float(cos.min()) >= 0.9999
        print(
            f"rows {ref_rows.shape[0]}  worst max_abs={worst_abs:.6f}  "
            f"min cosine={float(cos.min()):.6f}  {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            raise SystemExit("stage 1 FAILED: ONNX != kaldi")


# ---------------------------------------------------------------------------
# stage 2: wav -> text through the full python stack
# ---------------------------------------------------------------------------


def stage2_decode(model_dir: Path, wav: Path) -> str:
    print("== stage 2: wav -> text (full python stack) ==")

    # features + i-vector + AM + decode: the shipped recognizer stack
    samples, sr = load_audio(wav, read_meta(model_dir)["sample_rate"])
    rec = OfflineRecognizer(OnnxAm(model_dir))
    stream = rec.create_stream()
    stream.accept_waveform(sr, samples)
    stream.input_finished()
    print(f"audio: {samples.shape[0] / sr:.2f}s @ {sr} Hz, "
          f"features ({stream.num_frames_ready}, {read_meta(model_dir)['input_dim']})")
    print(f"ivector: dim={stream.ivector.shape[0]} (whole-utterance estimate)")

    rec.decode_stream(stream)
    text = rec.get_result(stream)
    if not text:
        raise SystemExit("stage 2 FAILED: decode produced nothing")
    print(f"decoded: {text!r}")
    return text


def vosk_baseline(model_dir: Path, wav: Path) -> str:
    """Reference text from the `vosk` package (same model, same wav).

    Runs in a SUBPROCESS: the vosk wheel bundles its own kaldi/openfst
    libraries; loading it in the same process as kaldi-hmm-gmm/kaldifst
    (which bundle theirs) segfaults on duplicate symbols.  Isolation is the
    safe pattern.
    """
    import subprocess

    script = f"""
import json
import soundfile as sf
import librosa
import vosk
vosk.SetLogLevel(-1)
audio, sr = sf.read({str(wav)!r}, dtype="float32", always_2d=True)
# feed vosk the SAME 16 kHz samples our stack uses (its own internal
# resampler would otherwise process different audio than ours)
if sr != 16000:
    audio = librosa.resample(audio[:, 0], orig_sr=sr, target_sr=16000)[:, None]
    sr = 16000
model = vosk.Model({str(model_dir)!r})
rec = vosk.KaldiRecognizer(model, sr)
pcm = (audio[:, 0] * {INT16_SCALE}).astype("int16").tobytes()
rec.AcceptWaveform(pcm)
print(json.loads(rec.FinalResult()).get("text", ""))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("vosk baseline failed:\n" + out.stderr[-1000:])
    return out.stdout.strip()


def main():
    p = argparse.ArgumentParser(
        description="Test an exported ONNX vosk model: AM parity vs kaldi "
        "(stage 1) and wav->text through the full python stack "
        "(stage 2)."
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="exported model dir (am/final.onnx, conf/, ivector/, graph/)",
    )
    p.add_argument("--wav", required=True, help="mono wav to decode")
    p.add_argument(
        "--kaldi-src",
        default=None,
        help="kaldi src dir for stage 1 (nnet3-compute etc.)",
    )
    p.add_argument(
        "--ref-model-dir",
        default=None,
        help="the ORIGINAL vosk model dir (has am/final.mdl): used "
        "as the kaldi reference in stage 1 and by the vosk "
        "baseline.  Required unless --skip-stage1 and the "
        "vosk package is absent.",
    )
    p.add_argument(
        "--expect",
        default=None,
        help="expected transcript -- the gate.  Compares "
        "whitespace-insensitively (the known text of the "
        "test wavs, e.g. zh.wav/en.wav are the JFK quote).",
    )
    p.add_argument(
        "--skip-stage1",
        action="store_true",
        help="skip the ONNX-vs-kaldi AM comparison",
    )
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args()
    model_dir, wav = Path(args.model_dir), Path(args.wav)

    ref = Path(args.ref_model_dir) if args.ref_model_dir else None

    if not args.skip_stage1:
        assert ref is not None, "stage 1 needs --ref-model-dir (original model)"
        kaldi_src = args.kaldi_src or default_kaldi_src()
        assert kaldi_src, "no kaldi for stage 1; pass --kaldi-src or --skip-stage1"
        stage1_am_parity(model_dir, ref, wav, kaldi_src, args.tolerance)

    text = stage2_decode(model_dir, wav)

    # The gate is the EXPECTED transcript (--expect).  The vosk baseline is
    # informational only: its streaming i-vector can degrade the end of an
    # utterance (observed: zh.wav via vosk ends '...国家 促使 呢' while this
    # stack decodes the correct '...国家 做 什么').
    try:
        if ref is not None:
            base = vosk_baseline(ref, wav)
            print(f"vosk baseline (informational): {base!r}")
    except Exception as e:  # noqa: BLE001 - baseline is best-effort
        print(f"NOTE: vosk baseline unavailable ({e})")

    if args.expect:
        ok = norm(text, drop_space=True) == norm(args.expect, drop_space=True)
        print("TEXT MATCH vs expect: " + ("OK" if ok else "FAIL"))
        if not ok:
            raise SystemExit(f"stage 2 FAILED: expected {args.expect!r}")
    else:
        print("NOTE: no --expect given; decoded text is informational only")
    print("ALL OK")


if __name__ == "__main__":
    main()
