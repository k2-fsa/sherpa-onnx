#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/upload-onnx/test.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Gate for an exported vosk ONNX model: decode the test wav with
# am/final.onnx AND am/final.int8.onnx through recognizer.py (which
# reproduces the vosk runtime, see its file comment) and compare each
# result with the `vosk` runtime on the ORIGINAL model.
#
# The reference is vosk's 1-best (SetMaxAlternatives(1)) with MFCC dither
# off, the whole wav fed in one AcceptWaveform call -- see
# compare_onnx.vosk_baseline.  recognizer.py computes exactly that, so the
# texts must be EQUAL (compared ignoring punctuation, case and
# whitespace).  The expected transcript is printed for information only:
# small models can genuinely decode a wav differently from its transcript,
# and then vosk does too.
#
# float32 decides the job.  The int8 decode is checked the same way, but
# a failure only SKIPS the int8 artifact (exit code 2) -- quantization can
# move a weak model's 1-best, and that should not block the float32
# export.  BOTH variants are always decoded and their results/errors go
# to the log, even when one of them is broken (missing file, decode error,
# empty output).
#
#   python3 test.py --model-dir ./onnx-model-cn --wav ./zh.wav \
#       --ref-model-dir ./vosk-model-small-cn-0.22 \
#       --expect '不要问你的国家能为你做什么，而要问你能为你的国家做什么。'
#
# Exit codes: 0 = both pass, 2 = float32 passed but int8 failed (skip
# int8), 1 = float32 failed (or the vosk reference is unavailable).

import argparse
import sys
import traceback
import unicodedata
from pathlib import Path

# scripts/vosk on sys.path for recognizer.py / compare_onnx.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare_onnx import vosk_baseline  # noqa: E402
from recognizer import OnnxAm, OfflineRecognizer, load_audio  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument(
        "--ref-model-dir",
        type=str,
        required=True,
        help="the ORIGINAL vosk model dir: its 1-best decode with the "
        "`vosk` package is the reference",
    )
    parser.add_argument(
        "--expect",
        type=str,
        default="",
        help="the known transcript of the wav (informational)",
    )
    return parser.parse_args()


def normalize(text):
    """case-folded NFC with all whitespace and punctuation removed, so
    only letters/digits/etc. are compared."""
    text = unicodedata.normalize("NFC", text).casefold()
    out = []
    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        out.append(ch)
    return "".join(out)


def decode(model_dir, wav, use_int8):
    rec = OfflineRecognizer(OnnxAm(model_dir, use_int8=use_int8))
    samples, sample_rate = load_audio(wav, rec.sample_rate)
    stream = rec.create_stream()
    stream.accept_waveform(sample_rate, samples)
    stream.input_finished()
    rec.decode_stream(stream)
    return rec.get_result(stream)


def main():
    args = get_args()

    try:
        ref = vosk_baseline(Path(args.ref_model_dir), Path(args.wav))
    except Exception as e:  # noqa: BLE001 - no reference, no gate
        raise SystemExit(f"test FAILED: vosk reference unavailable: {e}")
    print(f"[vosk] 1-best (dither off): {ref!r}")
    if args.expect:
        same = normalize(ref) == normalize(args.expect)
        print(
            "[vosk] vs the transcript: "
            + ("same" if same else f"differs (transcript: {args.expect})")
        )

    # both variants are always decoded and reported -- even when one of
    # them is broken (missing file, decode error, empty output), its
    # error belongs in the log.  The exit policy is applied afterwards:
    # float32 decides the job; an int8 failure only skips the int8
    # artifact.
    failed = {}
    for use_int8 in (False, True):
        variant = "int8" if use_int8 else "float32"
        try:
            text = decode(args.model_dir, args.wav, use_int8)
        except Exception:  # noqa: BLE001 - the error goes to the log
            traceback.print_exc()
            print(f"[{variant}] decode FAILED (see the traceback above)")
            failed[variant] = True
            continue
        ok = normalize(text) == normalize(ref)
        # repr: an empty decode shows up as '' instead of nothing
        print(f"[{variant}] decoded: {text!r}")
        print(f"[{variant}] MATCHES VOSK: {'OK' if ok else 'FAIL'}")
        failed[variant] = not ok

    if failed.get("float32"):
        raise SystemExit("test FAILED: float32 decode != the vosk runtime's")
    if failed.get("int8"):
        print("[int8] SKIPPED: the int8 model will not be shipped")
        raise SystemExit(2)

    print("ALL OK")


if __name__ == "__main__":
    main()
