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
#   │   │                         fields and `recurrent` added here)
#   │   └── transition_model.mdl  copied from the torch dir
#   ├── conf/  ivector/  graph/   copied unchanged
#   ├── log.md                    export record (markdown): gates,
#   │                             decimation analysis, toolchain -- the
#   │                             full story of this export
#   └── README
#
# Export contract:
#   inputs : features [B, T, input_dim] float32   (MFCC; dynamic B and T)
#            ivector  [B, T, ivector_dim] float32 (one i-vector per input
#                     frame, aligned with `features`: kaldi's online
#                     decoders -- and vosk -- give each chunk of frames
#                     its own i-vector; see torch_model.is_ivector_index)
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
# RECURRENT nets (LSTM; gigaspeech) are not decimated and are exported
# from the flat module (flatten.py), whose recurrent groups are
# TorchScript loop modules: the ONNX graph then holds a Loop with a
# dynamic trip count (tracing Nnet3Torch would unroll the per-row
# recursion to the dummy input's length).  See FlatFullRate.
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
from flatten import build_flat, remap_state_dict
from torch_model import Nnet3Torch, VoskAmWrapper, nnet_modulus

OPSET = 17
# time length of the dummy input used for tracing; T is a dynamic axis,
# so any length works (check_onnx_vs_torch runs several)
TRACE_NUM_FRAMES = 40


class FlatFullRate(torch.nn.Module):
    """Export unit for RECURRENT nets: the flat module (whose recurrent
    groups are TorchScript loop modules, so the ONNX graph has a Loop with
    a dynamic trip count -- tracing Nnet3Torch would unroll the per-row
    recursion to the dummy input's length) at full rate, rows ::factor,
    with the streaming state as an explicit input/output:

        (features [B, W, F], ivector [B, W, S], state [B, rows, dim])
            -> (log_probs [B, ceil(W/factor), num_pdfs], state_out)

    The window contract is the recognizer's: W = margin_left + n + R
    frames (margin_left = left_context rounded up to the row grid; n new
    frames); state_out is the state of the window starting n frames
    later, so shift = W - margin_left - R is derived from W (no extra
    input).  Zero state = the start of an utterance (== one-shot)."""

    def __init__(self, flat: torch.nn.Module, factor: int, margin: int):
        super().__init__()
        self.flat = flat
        self.factor = factor
        self.margin = margin  # margin_left + right_context

    def forward(
        self, features: torch.Tensor, ivector: torch.Tensor, state: torch.Tensor
    ):
        # a traced (dynamic) frame count, not a python int baked at trace
        shift = torch._shape_as_tensor(features)[1] - self.margin
        out, state_out = self.flat.forward_streaming(features, ivector, state, shift)
        return out[:, :: self.factor, :], state_out


def state_margin(net: Nnet3Torch, factor: int) -> int:
    """Frames of a window that are context, not new input: the left
    context rounded up to the row grid (the recognizer's window start)
    plus the right context."""
    return round_up(net.spec.left_context, factor) + net.spec.right_context


def export_unit(model: VoskAmWrapper, factor: int) -> torch.nn.Module:
    """What torch.onnx.export exports: the wrapper itself, or for nets
    with a time recurrence the (pre-traced) stateful flat module with
    scripted loops."""
    net = model.net
    if not net._recurrent:
        return model
    flat = build_flat(net.spec)
    flat.load_state_dict(remap_state_dict(net.state_dict(), net.spec))
    flat.eval()
    unit = FlatFullRate(flat, factor, state_margin(net, factor)).eval()
    # trace to a ScriptModule first: torch.onnx.export (TorchScript
    # exporter) refuses a scripted submodule inside a module it traces
    # itself ("not part of the active trace"), but exports an already
    # traced ScriptModule -- the scripted loops stay Loop nodes
    with torch.no_grad():
        return torch.jit.trace(unit, dummy_inputs(net), check_trace=False)


def dummy_inputs(net: Nnet3Torch) -> tuple:
    """Example inputs for tracing/export (T is a dynamic axis)."""
    spec = net.spec
    feats = torch.zeros(1, TRACE_NUM_FRAMES, spec.input_dim)
    iv = torch.zeros(1, TRACE_NUM_FRAMES, spec.ivector_dim)  # per frame
    if not net._recurrent:
        return feats, iv
    return feats, iv, net.initial_state(1)


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


def pin_state_out_shape(filename: Path, rows: int, dim: int) -> None:
    """state_out's shape is [batch, rows, dim], but shape inference leaves
    the concatenated feature dim symbolic; write the constants in (a
    runtime can then size the state from either end of the graph)."""
    import onnx

    model = onnx.load(str(filename))
    for out in model.graph.output:
        if out.name == "state_out":
            dims = out.type.tensor_type.shape.dim
            dims[1].dim_value = rows
            dims[2].dim_value = dim
    onnx.save(model, str(filename))


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


def input_affine_matmuls(filename: Path) -> list:
    """Names of the graph's input-affine MatMul nodes: the first learned
    layer(s), i.e. the MatMuls whose input comes straight from the
    model's features/ivector inputs (no other MatMul in between).

    The input affine is kept OUT of int8 quantization: its input spans
    raw MFCC + i-vector dimensions of very different scales, so its
    weights carry large per-channel dynamic ranges (30x on
    small-en-in-0.4) and one per-tensor int8 scale crushes the small
    channels.  Quantizing it alone flips most output rows (max_abs 3.1
    vs <= 0.14 for every other layer) and guts the decode; the rest of
    the net quantizes cleanly.  Measured on small-en-in-0.4,
    small-de-zamia-0.3, small-en-us-zamia-0.5, small-fr-pguyot-0.3:
    excluding the input affine restores the int8 decode to the float32
    one."""
    import onnx

    model = onnx.load(str(filename))
    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    def has_matmul_ancestor(tensor, seen):
        if tensor in seen:
            return False
        seen.add(tensor)
        node = producer.get(tensor)
        if node is None:
            return False  # graph input or initializer
        if node.op_type in ("MatMul", "Gemm"):
            return True
        return any(has_matmul_ancestor(i, seen) for i in node.input)

    return [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul" and not has_matmul_ancestor(node.input[0], set())
    ]


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


# decimation gate tolerance: see the docstring below
GATE_TOL = 1e-4


def decimation_gate(log: Log, model, net, factor: int) -> None:
    """Decimated == full-rate[::factor] -- the hard exactness gate,
    plus a measured speedup number.

    Tested on the real call contract: a context window around T rows
    (edge-replicated beyond the input, kaldi's edge rule).  Bare inputs
    are NOT equivalent at their edges -- the two paths clamp boundary
    gathers differently there -- but callers never keep edge rows (the
    window always carries L + R context), so the contract is the kept
    rows of a window.

    GATE_TOL: the two paths run different GEMM shapes (T rows vs T/3
    rows), so float32 accumulation noise of a few ULPs is expected --
    measured worst 3.8e-6 (en-us-0.21); a real decimation bug (wrong
    rows kept) shows up at O(0.01..10), so 1e-4 separates cleanly."""
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
        # one i-vector per frame (random per frame: stricter than the
        # per-chunk i-vectors of real decoding)
        iv = torch.from_numpy(
            (rng.standard_normal((1, feats.shape[1], iv_dim)) * 0.1).astype(np.float32)
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
            f"T={T:4d}: max_abs={d:.2e}   "
            f"{'bitwise equal' if d == 0.0 else 'OK' if d <= GATE_TOL else 'FAIL'}"
        )
    if worst > GATE_TOL:
        raise SystemExit(f"decimation gate FAILED: max_abs={worst}")

    # measured wall time, T=1000
    T = 1000
    feats = torch.randn(1, T, input_dim)
    iv = torch.randn(1, T, iv_dim) * 0.1

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
        iv = (rng.standard_normal((batch, T, ivector_dim)) * 0.1).astype(np.float32)

        with torch.no_grad():
            want = model(torch.from_numpy(feats), torch.from_numpy(iv)).numpy()
        feed = {"features": feats, "ivector": iv}
        if model.net._recurrent:  # one-shot == blank (zero) state
            feed["state"] = model.net.initial_state(batch).numpy()
        got = sess.run(["log_probs"], feed)[0]

        assert got.shape == want.shape, (got.shape, want.shape)
        # output rows (one per `factor` input frames): ceil(T/factor)
        assert got.shape[1] == (T + factor - 1) // factor, (got.shape, T)
        d = float(np.abs(want - got).max())
        # a time recurrence amplifies float noise (eager vs ORT kernels)
        # over long inputs -- strongly so on random features, which drive
        # the LSTM far outside its trained range (gigaspeech: 5e-7 at row
        # 0, O(1) by row ~150 of T=1000; real speech stays ~1e-3).  The
        # long run then only checks the dynamic shape; the strict numeric
        # gate is the short runs (and compare_onnx.py vs kaldi on speech).
        row_gate = gate and not (model.net._recurrent and T > 100)
        if d <= tolerance:
            status = "OK"
        elif not gate:
            status = "info"
        elif not row_gate:
            status = "info (recurrent net, long random input: shape only)"
        else:
            status = "FAIL"
        log.add(
            f"B={batch} T={T}: out {got.shape}  onnx-vs-torch "
            f"max_abs={d:.2e}  {status}"
        )
        if row_gate and d > tolerance:
            raise SystemExit(
                f"onnx/torch mismatch at B={batch} T={T} " f"(max_abs={d})"
            )


def _window(feats: np.ndarray, start: int, n: int) -> np.ndarray:
    """[B, n, F] rows start.. of feats [B, T, F], edge-replicated (the
    recognizer's get_frames)."""
    idx = np.clip(np.arange(start, start + n), 0, feats.shape[1] - 1)
    return feats[:, idx]


def check_onnx_streaming(
    onnx_path: Path,
    net: Nnet3Torch,
    factor: int,
    tolerance: float,
    log: Log,
    gate: bool = True,
) -> None:
    """The streaming-state contract of a recurrent model.

    1. handover: a window fed a NON-blank state (torch's state after a
       previous window) gives the same log-probs and state_out in
       onnxruntime as in torch (Nnet3Torch.forward_streaming), B=2;
    2. chaining: the recognizer's windows (margin + chunk + R frames,
       state carried) reproduce the one-shot run, chunk 30.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    spec = net.spec
    gl = round_up(spec.left_context, factor)
    R = spec.right_context
    rng = np.random.default_rng(4321)
    feats = rng.standard_normal((2, 300, spec.input_dim)).astype(np.float32)
    iv = (rng.standard_normal((2, 300, spec.ivector_dim)) * 0.1).astype(np.float32)
    chunk = 10 * factor

    # 1. handover
    w1 = _window(feats, -gl, gl + chunk + R)
    w2 = _window(feats, chunk - gl, gl + chunk + R)
    v1 = _window(iv, -gl, gl + chunk + R)
    v2 = _window(iv, chunk - gl, gl + chunk + R)
    with torch.no_grad():
        _, s1 = net.forward_streaming(
            torch.from_numpy(w1), torch.from_numpy(v1), net.initial_state(2), chunk
        )
        want, want_s = net.forward_streaming(
            torch.from_numpy(w2), torch.from_numpy(v2), s1, chunk
        )
    got, got_s = sess.run(
        ["log_probs", "state_out"],
        {"features": w2, "ivector": v2, "state": s1.numpy()},
    )
    d = float(np.abs(want[:, ::factor].numpy() - got).max())
    ds = float(np.abs(want_s.numpy() - got_s).max())
    ok1 = d <= tolerance and ds <= tolerance
    log.add(
        f"state handover (B=2, chunk {chunk}): log_probs max_abs={d:.2e} "
        f"state_out max_abs={ds:.2e}  "
        + ("OK" if ok1 else ("FAIL" if gate else "info"))
    )

    # 2. chained windows vs one-shot (onnxruntime on both sides)
    T = feats.shape[1]
    n_rows = -(-T // factor)
    zero = net.initial_state(2).numpy()
    one = sess.run(
        ["log_probs"],
        {
            "features": _window(feats, -gl, gl + n_rows * factor + R),
            "ivector": _window(iv, -gl, gl + n_rows * factor + R),
            "state": zero,
        },
    )[0][:, gl // factor : gl // factor + n_rows]
    rows_out, state, t = [], zero, 0
    while t < T:
        rows = min(chunk // factor, -(-(T - t) // factor))
        r, state = sess.run(
            ["log_probs", "state_out"],
            {
                "features": _window(feats, t - gl, gl + rows * factor + R),
                "ivector": _window(iv, t - gl, gl + rows * factor + R),
                "state": state,
            },
        )
        rows_out.append(r[:, gl // factor : gl // factor + rows])
        t += chunk
    streamed = np.concatenate(rows_out, axis=1)
    row_max = np.abs(streamed - one).max(axis=(0, 2))
    argmax_ok = float((streamed.argmax(-1) == one.argmax(-1)).mean())
    ok2 = float(np.median(row_max)) <= tolerance and argmax_ok >= 0.99
    log.add(
        f"chained windows vs one-shot (B=2, T={T}, chunk {chunk}): "
        f"max_abs={row_max.max():.2e} median={np.median(row_max):.2e} "
        f"argmax={argmax_ok:.3f}  " + ("OK" if ok2 else ("FAIL" if gate else "info"))
    )
    if gate and not (ok1 and ok2):
        raise SystemExit("streaming state check FAILED")


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
    recurrent = bool(model.net._recurrent)
    dynamic_axes = {
        "features": {0: "batch", 1: "time"},  # variable shape
        "ivector": {0: "batch", 1: "time"},  # one per frame
        "log_probs": {0: "batch", 1: "time_out"},
    }
    if recurrent:  # streaming state in/out (see FlatFullRate)
        dynamic_axes["state"] = {0: "batch"}
        dynamic_axes["state_out"] = {0: "batch"}
    torch.onnx.export(
        export_unit(model, factor),
        dummy_inputs(model.net),
        str(onnx_path),
        input_names=["features", "ivector"] + (["state"] if recurrent else []),
        output_names=["log_probs"] + (["state_out"] if recurrent else []),
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,  # the TorchScript exporter; dynamic_axes + dynamo
        #               warns and can hit constraint errors
    )
    if recurrent:
        pin_state_out_shape(onnx_path, model.net.state_rows, model.net.state_dim)
    log.add(f"`wrote {onnx_path.name}: opset {OPSET}, dynamic batch+time`")
    if recurrent:
        log.add(
            f"`streaming state: state [B, {model.net.state_rows}, "
            f"{model.net.state_dim}] -> state_out "
            f"({', '.join(n for n, _o, _d in model.net.state_layout)})`"
        )
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
    # 1 = the net has a time recurrence (LSTM): its rows depend on how far
    # back the fed input reaches, and float noise can be amplified
    meta["recurrent"] = 1 if model.net._recurrent else 0
    # streaming state shape [B, state_rows, state_dim] (0, 0: no state)
    meta["state_rows"] = model.net.state_rows
    meta["state_dim"] = model.net.state_dim
    # kaldi's Nnet::Modulus(): the looped decodable's frames_per_chunk is
    # rounded up to a multiple of it (and of the subsampling factor)
    meta["modulus"] = nnet_modulus(model.net.spec.nodes)
    add_meta_data(onnx_path, meta)
    (out_dir / "am" / "final.onnx.data").unlink(missing_ok=True)
    # keep the on-disk model_meta.json in sync with the embedded metadata
    meta_file = out_dir / "am" / "model_meta.json"
    meta_file.write_text(
        json.dumps(
            {
                **meta_in,
                **{
                    k: meta[k]
                    for k in (
                        "decimated_zone",
                        "compute_ratio",
                        "recurrent",
                        "state_rows",
                        "state_dim",
                        "modulus",
                    )
                },
            },
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
    if model.net._recurrent:
        check_onnx_streaming(onnx_path, model.net, factor, args.tolerance, log)
    log.only("```")
    log.add("")

    # ---- 5. int8 model (dynamic quantization; QInt8 weights) ----------
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out_dir / "am" / "final.int8.onnx"
    # keep the input affine in float32 -- its per-channel weight ranges
    # (raw MFCC + i-vector scales) do not survive one per-tensor int8
    # scale; quantizing it guts the decode (see input_affine_matmuls)
    skip = input_affine_matmuls(onnx_path)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=skip,
    )
    add_meta_data(int8_path, meta)  # keep metadata with the model
    log.add("## int8 (quantize_dynamic, QInt8 weights)")
    log.add("")
    log.only("```")
    log.add(f"wrote {int8_path.name} (QInt8)")
    log.add(f"kept in float32 (input affine): {', '.join(skip) if skip else '(none)'}")
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
    if model.net._recurrent:
        check_onnx_streaming(
            int8_path, model.net, factor, tolerance=1.0, log=log, gate=False
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
