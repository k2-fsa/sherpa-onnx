#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/describe_onnx.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Print the input/output signature of an exported vosk ONNX model
# (export_onnx.py output) as a markdown table: every graph input and
# output with its name, shape and dtype.  This is what the info pages
# (upload-onnx/info.md.in) show and what you need to call the model.
#
#   python3 describe_onnx.py --model-dir ./onnx-model-cn
#   python3 describe_onnx.py --model-dir ./onnx-model-cn --onnx final.int8.onnx
#
# Shapes keep the dynamic dim names of the graph (`batch`, `time`, ...),
# and the note under the table is derived from am/model_meta.json.

import argparse
import json
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(
        description="ONNX input/output signature of an exported vosk model"
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="exported model dir (am/final.onnx, am/model_meta.json, ...)",
    )
    p.add_argument(
        "--onnx",
        default="final.onnx",
        help="which am/<file>.onnx to read (default final.onnx)",
    )
    return p.parse_args()


# onnxruntime's element type name -> the numpy/ONNX dtype name
DTYPES = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int16)": "int16",
    "tensor(int8)": "int8",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


def dtype_name(ort_type):
    if ort_type in DTYPES:
        return DTYPES[ort_type]
    return ort_type.removeprefix("tensor(").removesuffix(")")  # unknown


def shape_name(shape):
    """[batch, time, 40]: dynamic dims as their graph names, fixed as
    numbers (a symbolic dim is a str, a fixed one an int)."""
    return "[" + ", ".join(str(d) for d in shape) + "]"


def signature(sess):
    """[(input/output, name, shape, dtype)] for one ONNX session."""
    out = []
    for kind, values in (("input", sess.get_inputs()), ("output", sess.get_outputs())):
        for v in values:
            out.append((kind, v.name, shape_name(v.shape), dtype_name(v.type)))
    return out


def note(model_dir: Path, table) -> list:
    """The bullets under the table: what the shapes mean, from
    model_meta.json."""
    meta = json.loads((model_dir / "am" / "model_meta.json").read_text())
    factor = int(meta["frame_subsampling_factor"])
    notes = []
    names = {name for _k, name, _s, _d in table}
    if "features" in names:
        notes.append(
            "`features` [batch, time, {}]: the MFCC frames of `conf/mfcc.conf` "
            "({} Hz, int16 scale).  `time` is dynamic: any number of frames.".format(
                meta["input_dim"], meta["sample_rate"]
            )
        )
    if "ivector" in names:
        notes.append(
            "`ivector` [batch, time, {}]: ONE i-vector per input frame, aligned "
            "with `features`.  kaldi's online decoders (and the vosk runtime) "
            "compute the network in chunks of `--frames-per-chunk` frames "
            "(24) and give each chunk its own i-vector -- repeat each chunk's "
            "i-vector over its frames.  `scripts/vosk/recognizer.py` does "
            "this (`IvectorChunker`).".format(meta["ivector_dim"])
        )
    if "state" in names:
        notes.append(
            "`state` [batch, {}, {}]: the LSTM state carried from one decode "
            "window to the next (zeros = the start of an utterance, or a "
            "one-shot run); `state_out` is the state for the window starting "
            "`shift` frames later.".format(meta["state_rows"], meta["state_dim"])
        )
    if "log_probs" in names:
        notes.append(
            "`log_probs` [batch, time_out, {}]: one row per {} input frames, "
            "row k anchored at input frame k x {} ({} ms per row), so "
            "time_out = ceil(time / {}).".format(
                meta["num_pdfs"], factor, factor, 10 * factor, factor
            )
        )
    notes.append(
        "The model is pure: the caller supplies the receptive field "
        "({} frames left, {} right) around the frames it wants; edge frames "
        "are replicated (kaldi's edge rule).".format(
            meta["left_context"], meta["right_context"]
        )
    )
    return notes


def table_md(table):
    lines = ["| input/output | name | shape | dtype |", "|---|---|---|---|"]
    lines += [f"| {k} | `{n}` | `{s}` | {d} |" for k, n, s, d in table]
    return lines


def main():
    args = get_args()
    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "am" / args.onnx
    if not onnx_path.is_file():
        raise SystemExit(f"{onnx_path} not found")

    import onnxruntime as ort

    table = signature(
        ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    )

    # the int8 variant must be callable the same way
    int8 = model_dir / "am" / "final.int8.onnx"
    int8_same = None
    if int8.is_file():
        int8_same = (
            signature(
                ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
            )
            == table
        )

    lines = table_md(table)
    files = f"`am/{args.onnx}`"
    if int8.is_file():
        if int8_same:
            files += " and `am/final.int8.onnx` (same signature)"
        else:
            files += " (`am/final.int8.onnx` differs -- see its own table)"
    print(files + ":")
    print()
    print("\n".join(lines))
    print()
    for n in note(model_dir, table):
        print(f"- {n}")
    if int8.is_file() and not int8_same:
        print()
        print("`am/final.int8.onnx`:")
        print()
        print(
            "\n".join(
                table_md(
                    signature(
                        ort.InferenceSession(
                            str(int8), providers=["CPUExecutionProvider"]
                        )
                    )
                )
            )
        )


if __name__ == "__main__":
    main()
