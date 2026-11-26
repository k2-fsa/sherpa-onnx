#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/export_onnx.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Export a PyTorch model dir (from export_torch.py) to an ONNX model dir —
# the final stage of the model conversion:
#
#     torch model dir  --export_onnx.py-->  onnx model dir
#
# The output is the ORIGINAL model directory layout with am/final.onnx in
# place of am/final.mdl:
#
#   <out_dir>/
#   ├── am/
#   │   ├── final.onnx            float32 AM (single file, no external
#   │   │                         data; runtime metadata embedded)
#   │   ├── final.int8.onnx       dynamically quantized AM (QInt8 weights)
#   │   ├── model_meta.json       copied from the torch dir (+ decimation
#   │   │                         fields added here)
#   │   └── transition_model.mdl  copied from the torch dir
#   ├── conf/  ivector/  graph/   copied unchanged
#   ├── log.md                    export record (markdown): gates,
#   │                             decimation analysis, toolchain -- the
#   │                             full story of this export
#   └── README
#
# Export contract:
#   inputs : features [B, T, input_dim] float32   (MFCC; dynamic B and T)
#            ivector  [B, ivector_dim] float32    (broadcast over time)
#   output : log_probs [B, ceil(T/factor), num_pdfs] float32
#
# factor comes from model_meta.json (frame_subsampling_factor; kaldi
# anchors decoder row k at input frame k*factor -- nnet-compile-looped.cc
# -- the row grid always starts at input frame 0).  The subsampling is computed *inside*
# the graph on the decoder's row lattice: the decimated zone evaluates only
# the rows the decoder reads -- kaldi-style row sparsity (kaldi's looped
# computation request asks for exactly those rows, see
# torch_model.DecimationPlan) -- identical values at ~1/factor the rows in
# that zone.  This script gates decimated against the full-rate computation
# and records the whole analysis in <out_dir>/log.md.
#
# VARIABLE SHAPE is a hard requirement: both B and T are
# dynamic_axes, and the check below runs onnxruntime at several different
# sequence lengths and compares against torch at each.
#
# METADATA is embedded in both .onnx files as onnx metadata_props (same
# convention as ../whisper/export_onnx.py): the key-
# value pairs of am/model_meta.json, readable by the C++ runtime and by
# onnxruntime's session.get_modelmeta().custom_metadata_map -- no sidecar
# needed at inference time.
#
# Usage:
#   python3 export_onnx.py --in-dir ./torch-model-cn \
#       --out-dir ./onnx-model-cn

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


from export_torch import copy_model_tree
from recognizer import round_up
from torch_model import Nnet3Torch, VoskAmWrapper

OPSET = 17
# time length of the dummy input used for tracing; T is a dynamic axis,
# so any length works (check_onnx_vs_torch runs several)
TRACE_NUM_FRAMES = 40


def _md_plain(s: str) -> str:
    """Strip markdown decorations for the stdout copy of the record."""
    s = re.sub(r"^#{1,6}\s+", "", s)
    s = re.sub(r"^\s*[-*]\s+", "", s)
    return s.replace("**", "").replace("`", "")


class Log:
    """The export record: markdown in <out_dir>/log.md.

    Every line goes to the file; `echo` lines also print to stdout (with
    the markdown decorations stripped).  Structural markdown (headings,
    code fences) is file-only.  `write(result)` closes the record -- also
    on failure, so the file always describes what happened."""

    def __init__(self, out_dir: Path):
        self.lines = []
        self.out_dir = out_dir

    def add(self, s: str = "", echo: bool = True):
        self.lines.append(s)
        if echo:
            print(_md_plain(s))

    def only(self, s: str = ""):
        self.lines.append(s)

    def write(self, result: str):
        self.lines.append("")
        self.lines.append(f"**result: {result}**")
        (self.out_dir / "log.md").write_text("\n".join(self.lines) + "\n")


def get_args():
    p = argparse.ArgumentParser(
        description="Export a torch model dir (export_torch.py output) to an "
        "ONNX model dir (--in-dir / --out-dir)."
    )
    p.add_argument(
        "--in-dir", required=True, help="torch model dir (contains am/am.pt)"
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="output model dir: original layout with am/final.onnx",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="max abs diff between onnxruntime and torch (kernel "
        "float noise between eager and ORT matmuls is ~1e-5; "
        "the stricter gate is compare_onnx.py vs kaldi)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="print the per-layer decimation table to stdout too " "(always in log.md)",
    )
    return p.parse_args()


def load_torch_model(am_pt: Path, factor: int):
    """Rebuild the net from the saved dict and wrap it.

    Returns (VoskAmWrapper, meta-ish info from the args dict).  The wrapper
    computes rows ::factor on the decoder's row lattice (kaldi-style
    row sparsity; factor comes from model_meta.json, never assumed).
    """
    payload = torch.load(am_pt, map_location="cpu", weights_only=False)
    net = Nnet3Torch.from_export_args(payload["args"])
    net.load_state_dict(payload["state_dict"])
    net.eval()
    model = VoskAmWrapper(net, frame_subsampling_factor=factor)
    model.eval()
    return model, payload["args"]


def add_meta_data(filename: Path, meta_data: dict) -> None:
    """Embed key-value metadata into an ONNX model (in-place).

    Same convention as ../whisper/export_onnx.py: the
    pairs land in the model's metadata_props and are read back by the C++
    runtime and by onnxruntime (session.get_modelmeta().
    custom_metadata_map).  Saving folds any external tensors into the one
    .onnx file.
    """
    import onnx

    model = onnx.load(str(filename))
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, str(filename))  # single file, no external data


# ---------------------------------------------------------------------------
# decimation analysis + gate
# ---------------------------------------------------------------------------


def decimation_report(log: Log, model, net, factor: int, verbose: bool):
    """Zone summary (stdout + log), per-layer table (log; stdout if
    --verbose).  Evidence, not claims: offsets and needed rows per node."""
    plan = model.plan
    comps = {c.name: c for c in net.spec.components}
    nodes = {n.name: n for n in net.spec.nodes}

    def offsets_str(name):
        c = comps.get(nodes[name].component)
        if c is None or c.type != "TdnnComponent":
            return "[0]"
        return "[" + ", ".join(str(o) for o in c.params["TimeOffsets"]) + "]"

    def where(zone):
        names = [n for n in plan.nodes if plan.zones[n] == zone]
        if not names:
            return "-"
        if zone == "transition":
            return ", ".join(names)
        return f"{names[0]} .. {names[-1]}"

    log.add(f"## decimation analysis (output rows 0::{factor})")
    log.add("")
    log.only("```")
    log.add(f"{'zone':<12} {'nodes':>5} {'MAC/row':>10} {'rows kept':>10}   where")
    for zone in ("dense", "transition", "decimated"):
        names = [n for n in plan.nodes if plan.zones[n] == zone]
        mac = sum(plan.mac[n] for n in names)
        kept = sorted({plan.needed_rows(n) for n in names})
        log.add(
            f"{zone:<12} {len(names):>5} {mac:>10,.0f} "
            f"{','.join(kept) if kept else '-':>10}   {where(zone)}"
        )
    log.add(
        f"cut: {' + '.join(plan.boundary) if plan.boundary else 'none'} "
        f"-> decimated grid [rows 0::{factor}]"
    )
    total = sum(plan.mac.values())
    ratio = plan.compute_ratio
    log.add(
        f"compute: {total / 1e6:.2f}M x T -> {total * ratio / 1e6:.2f}M x T MAC"
        f"   ({(1 - ratio) * 100:.0f}% saved, {1 / ratio:.1f}x AM)"
    )
    if plan.transition_extra_ratio > 0:
        log.add(
            f"note: transition rows not exploited in this export "
            f"(exact row masks would save +{plan.transition_extra_ratio * 100:.0f}%)"
        )
    log.only("```")
    log.add("")

    # per-layer evidence: file always, stdout with --verbose
    log.only("per-layer evidence (component offsets and needed rows):")
    log.only("")
    table = [
        f"{'component':<32} {'offsets':<14} {'zone':<12} "
        f"{'MAC/row':>10} {'needed rows':>12}"
    ]
    for n in plan.nodes:
        if plan.zones[n] == "input":
            continue
        table.append(
            f"{n:<32} {offsets_str(n):<14} {plan.zones[n]:<12} "
            f"{plan.mac[n]:>10,.0f} {plan.needed_rows(n):>12}"
        )
    log.only("```")
    for line in table:
        log.add(line, echo=verbose)
    log.only("```")
    log.only("")


def decimation_gate(log: Log, model, net, factor: int) -> None:
    """Decimated == full-rate[::factor] -- the hard exactness gate,
    plus a measured speedup number.

    Tested on the real call contract: a context window around T rows
    (edge-replicated beyond the input, kaldi's edge rule).  Bare inputs
    are NOT equivalent at their edges -- the two paths clamp boundary
    gathers differently there -- but callers never keep edge rows (the
    window always carries L + R context), so the contract is the kept
    rows of a window."""
    log.add(f"## decimation gate (decimated == full-rate[::{factor}])")
    log.add("")
    log.only("```")
    worst = 0.0
    input_dim = model.net.spec.input_dim
    iv_dim = model.net.spec.ivector_dim
    grid_left = round_up(model.net.spec.left_context, factor)
    right = model.net.spec.right_context
    rng = np.random.default_rng(7)
    for T in (100, 1000, 33):
        feats = np.pad(
            rng.standard_normal((1, T, input_dim)).astype(np.float32),
            ((0, 0), (grid_left, right), (0, 0)),
            mode="edge",
        )
        feats = torch.from_numpy(feats)
        iv = torch.from_numpy(
            (rng.standard_normal((1, iv_dim)) * 0.1).astype(np.float32)
        )
        with torch.no_grad():
            want = net(feats, iv)[
                :, grid_left : grid_left + T : factor
            ].numpy()  # full-rate rows of the real frames
            got = model(feats, iv).numpy()[
                :, grid_left // factor :
            ]  # the exported path
        got = got[:, : want.shape[1]]
        d = float(np.abs(got - want).max())
        worst = max(worst, d)
        log.add(
            f"T={T:4d}: max_abs={d:.2e}   " f"{'bitwise equal' if d == 0.0 else 'OK'}"
        )
    if worst > 1e-6:
        raise SystemExit(f"decimation gate FAILED: max_abs={worst}")

    # measured wall time, T=1000
    T = 1000
    feats = torch.randn(1, T, input_dim)
    iv = torch.randn(1, iv_dim) * 0.1

    def bench(fn, reps=20):
        with torch.no_grad():
            fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
        return (time.perf_counter() - t0) / reps * 1000

    with torch.no_grad():
        t_dec = bench(lambda: model(feats, iv))
        t_full = bench(lambda: net(feats, iv)[:, ::factor, :])
    log.add(
        f"timing: full-rate {t_full:.1f} ms vs decimated {t_dec:.1f} ms "
        f"(T={T}, CPU, x{t_full / t_dec:.1f})"
    )
    log.only("```")
    log.add("")


def check_onnx_vs_torch(
    onnx_path: Path,
    model: VoskAmWrapper,
    args: dict,
    factor: int,
    tolerance: float,
    log: Log,
    gate: bool = True,
) -> None:
    """Run onnxruntime at several sequence lengths and compare with torch.

    This is the variable-shape gate: dynamic B and T must both work, and the
    outputs must match the eager model at each length.  ``gate=False`` is
    informational only (used for the int8 model, whose quantization noise
    is expected).
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_dim, ivector_dim = args["input_dim"], args["ivector_dim"]

    rng = np.random.default_rng(1234)
    for batch, T in ((1, 100), (2, 1000), (1, 33)):  # incl. odd length
        feats = rng.standard_normal((batch, T, input_dim)).astype(np.float32)
        iv = (rng.standard_normal((batch, ivector_dim)) * 0.1).astype(np.float32)

        with torch.no_grad():
            want = model(torch.from_numpy(feats), torch.from_numpy(iv)).numpy()
        got = sess.run(["log_probs"], {"features": feats, "ivector": iv})[0]

        assert got.shape == want.shape, (got.shape, want.shape)
        # output rows (one per `factor` input frames): ceil(T/factor)
        assert got.shape[1] == (T + factor - 1) // factor, (got.shape, T)
        d = float(np.abs(want - got).max())
        status = "OK" if d <= tolerance else ("info" if not gate else "FAIL")
        log.add(
            f"B={batch} T={T}: out {got.shape}  onnx-vs-torch "
            f"max_abs={d:.2e}  {status}"
        )
        if gate and d > tolerance:
            raise SystemExit(
                f"onnx/torch mismatch at B={batch} T={T} " f"(max_abs={d})"
            )


def main():
    args = get_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    am_pt = in_dir / "am" / "am.pt"
    assert am_pt.is_file(), f"missing {am_pt} (run export_torch.py first)"

    # the subsampling contract: read from the model's own metadata, never
    # assumed (factor is not always 3)
    meta_in = json.loads((in_dir / "am" / "model_meta.json").read_text())
    factor = int(meta_in["frame_subsampling_factor"])

    out_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out_dir)
    log.only("# export record — vosk AM -> ONNX")
    log.only("")
    log.add(f"- **date:** {datetime.now():%Y-%m-%d %H:%M}")
    log.add(f"- **source:** {meta_in.get('source_model', '?')}  <-  {am_pt}")
    log.add(f"- **toolchain:** torch {torch.__version__}, opset {OPSET}")

    try:
        _export(args, in_dir, out_dir, meta_in, factor, log)
    except BaseException as e:
        log.write(f"FAILED ({type(e).__name__}: {e})")
        raise
    log.write("OK")


def _export(args, in_dir, out_dir, meta_in, factor, log):
    am_pt = in_dir / "am" / "am.pt"

    # ---- 1. rebuild the model from the dict ---------------------------
    model, model_args = load_torch_model(am_pt, factor)
    log.add(
        f"- **model:** input_dim={model_args['input_dim']} "
        f"ivector_dim={model_args['ivector_dim']} "
        f"num_pdfs={model_args['output_dim']}"
    )
    log.add(
        f"- **subsampling:** frame_subsampling_factor={factor} "
        f"left/right={meta_in['left_context']}/{meta_in['right_context']}"
    )
    log.add("")

    # ---- 2. copy the tree, then export am/final.onnx ------------------
    copy_model_tree(in_dir, out_dir)

    onnx_path = out_dir / "am" / "final.onnx"
    feats_dummy = torch.zeros(1, TRACE_NUM_FRAMES, model_args["input_dim"])
    iv_dummy = torch.zeros(1, model_args["ivector_dim"])
    torch.onnx.export(
        model,
        (feats_dummy, iv_dummy),
        str(onnx_path),
        input_names=["features", "ivector"],
        output_names=["log_probs"],
        dynamic_axes={
            "features": {0: "batch", 1: "time"},  # variable shape
            "ivector": {0: "batch"},
            "log_probs": {0: "batch", 1: "time_out"},
        },
        opset_version=OPSET,
    )
    log.add(f"`wrote {onnx_path.name}: opset {OPSET}, dynamic batch+time`")
    log.add("")

    # ---- 3. decimation analysis --------------------------------------
    # factor == 1 -> VoskAmWrapper runs the plain full-rate path with
    # plan=None: nothing to analyze or gate
    plan = model.plan
    if plan is not None:
        decimation_report(log, model, model.net, factor, args.verbose)
        decimation_gate(log, model, model.net, factor)
    else:
        log.add(f"## decimation analysis (output rows 0::{factor})")
        log.add("")
        log.add(
            f"frame_subsampling_factor={factor}: no decimation " f"(compute ratio 1.0)"
        )
        log.add("")

    # metadata: model_meta.json pairs + the decimation record
    meta = {k: v for k, v in meta_in.items() if k != "components"}
    meta["decimated_zone"] = (
        "+".join(plan.boundary) if plan is not None and plan.boundary else "none"
    )
    meta["compute_ratio"] = f"{plan.compute_ratio:.2f}" if plan is not None else "1.0"
    add_meta_data(onnx_path, meta)
    (out_dir / "am" / "final.onnx.data").unlink(missing_ok=True)
    # keep the on-disk model_meta.json in sync with the embedded metadata
    meta_file = out_dir / "am" / "model_meta.json"
    meta_file.write_text(
        json.dumps(
            {**meta_in, **{k: meta[k] for k in ("decimated_zone", "compute_ratio")}},
            indent=2,
        )
        + "\n"
    )
    log.add(f"`embedded metadata: {', '.join(meta)}`")
    log.add("")

    # ---- 4. variable-shape + numeric check (float32, gated) -----------
    log.add("## checks (onnxruntime vs torch)")
    log.add("")
    log.only("```")
    check_onnx_vs_torch(onnx_path, model, model_args, factor, args.tolerance, log)
    log.only("```")
    log.add("")

    # ---- 5. int8 model (dynamic quantization; QInt8 weights) ----------
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out_dir / "am" / "final.int8.onnx"
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    add_meta_data(int8_path, meta)  # keep metadata with the model
    log.add("## int8 (quantize_dynamic, QInt8 weights)")
    log.add("")
    log.only("```")
    log.add(f"wrote {int8_path.name} (QInt8)")
    log.add("int8 check (onnxruntime vs torch, informational):")
    check_onnx_vs_torch(
        int8_path,
        model,
        model_args,
        factor,
        tolerance=1.0,
        log=log,
        gate=False,
    )
    log.only("```")
    log.add("")

    # ---- 6. the torch side is not part of the final artifact ----------
    # (am_flat.py is generated from am.pt -- without it, it cannot load)
    (out_dir / "am" / "am.pt").unlink()
    flat = out_dir / "am" / "am_flat.py"
    if flat.is_file():
        flat.unlink()
    log.add("## files")
    log.add("")
    log.only("```")
    for p in (
        "am/final.onnx",
        "am/final.int8.onnx",
        "am/model_meta.json",
        "am/transition_model.mdl",
    ):
        f = out_dir / p
        if f.is_file():
            log.add(f"{p:<26} {f.stat().st_size / 1e6:6.2f} MB")
    log.add(f"{'log.md':<26} (this file)")
    log.only("```")
    log.add("")
    log.add(f"`done: {out_dir}`")


if __name__ == "__main__":
    main()
