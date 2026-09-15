#!/usr/bin/env python3
"""Weight-only per-channel INT8 quantization for Nabra-82M ONNX.

Strategy (mirrors the repo's own FP16 trick, generalized):
  - Replace each large float initializer with an int8 copy + per-output-channel scales.
  - Insert a DequantizeLinear node before each consumer so the graph computes in
    FP32 with the same kernels as before. No ConvInteger/QLinearConv anywhere,
    so none of the ARM64 integer-conv regression applies.
  - Small tensors (< --min-bytes) stay FP32: biases, norms, adain tables.

LSTM note: W is [dir, 4*hidden, input] -> per-axis along axis=1.
"""
import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper


def find_all_nodes(graph):
    """Yield (node, owner_graph) for the top graph and every subgraph."""
    out = [(n, graph) for n in graph.node]
    for n in graph.node:
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                out.extend(find_all_nodes(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.gs:
                    out.extend(find_all_nodes(sg))
    return out


# Which input slot of each op holds the weight (vs bias etc.)
WEIGHT_SLOTS = {"Conv": {1}, "ConvTranspose": {1}, "Gemm": {1}, "MatMul": {0, 1}, "LSTM": {1, 2}}
# Ops that pass a tensor through unchanged: safe consumers of an already-dequantized weight.
# ALBERT weight-sharing forwards one tensor through many Identity nodes;
# embedding tables feed Gather directly.
TRANSPARENT_SLOTS = {"Identity": {0}, "Gather": {0}}


def consumer_axis(node, slot):
    """Quantization axis so dequantized values line up with the op's output channels."""
    op = node.op_type
    if op == "Conv":
        return 0            # [O, I/group, k...]
    if op == "ConvTranspose":
        return 1            # [I, O/group, k...]: output channels on dim 1!
    if op == "Gemm":
        # B stored [out, in] when transB=1 (PyTorch nn.Linear default), else [in, out]
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        return 0 if trans_b == 1 else -1
    if op == "MatMul":
        return -1           # both operands contribute output features along last dim (2D)
    if op == "Gather":
        return 0            # embedding table: one channel per vocabulary row
    if op == "LSTM":
        return 1            # [dir, 4H, x]
    raise ValueError(f"unexpected consumer {op}")


def quantize_tensor(arr, axis, scale_search=False):
    """Symmetric per-channel int8. Returns (q_int8, scales_float32).

    scale_search: try slightly smaller scales and keep whichever minimizes
    squared error (max-abs scales waste range when outliers are rare).
    """
    if axis < 0:
        axis += arr.ndim
    if arr.ndim <= 1:
        m = np.max(np.abs(arr.astype(np.float64)))
        scale = np.float32(m / 127.0 if m > 0 else 1e-12)
        q = np.clip(np.rint(arr / scale), -127, 127).astype(np.int8)
        return q, np.array([scale], dtype=np.float32)
    axis_n = axis % arr.ndim   # normalize -1 -> last dim; otherwise red_dims collapses to scalar!
    red_dims = tuple(d for d in range(arr.ndim) if d != axis_n)
    m = np.max(np.abs(arr.astype(np.float64)), axis=red_dims, keepdims=True)
    scales = (m / 127.0).astype(np.float32)
    if scale_search:
        a64 = arr.astype(np.float64)
        best_err, best_scales = None, scales
        for f in np.linspace(0.78, 1.0, 12):
            s = np.where(scales == 0, np.float32(1e-12),
                         (scales.astype(np.float64) * f).astype(np.float32))
            deq = np.clip(np.rint(a64 / s.astype(np.float64)), -127, 127) * s.astype(np.float64)
            err = ((deq - a64) ** 2).sum()
            if best_err is None or err < best_err:
                best_err, best_scales = err, s
        scales = best_scales
    scales_safe = np.where(scales == 0, np.float32(1e-12), scales)
    q = np.clip(np.rint(arr / scales_safe), -127, 127).astype(np.int8)
    return q, scales.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/nabra_fp32.onnx")
    ap.add_argument("--dst", default="models/nabra_int8.onnx")
    ap.add_argument("--min-bytes", type=int, default=16384,
                    help="quantize float initializers at least this many bytes")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; only tensors matching are quantized")
    ap.add_argument("--keep-fp16", default=None,
                    help="comma-separated substrings; matching tensors become FP16-stored "
                         "+ Cast-to-FP32 (repo's fp16 trick) instead of int8")
    ap.add_argument("--fp16-small", action="store_true",
                    help="after main pass, convert every remaining float initializer "
                         "(biases, norms) to FP16 storage + Cast")
    ap.add_argument("--lstm-r-fp16", action="store_true",
                    help="keep LSTM recurrent matrices (slot 2) FP16 even when the "
                         "rest of their tensor family quantizes")
    ap.add_argument("--scale-search", action="store_true",
                    help="MSE-optimize each scale instead of plain max-abs")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated substrings; matching tensors stay FP32")
    args = ap.parse_args()

    print(f"Loading {args.src} ...")
    model = onnx.load(args.src)
    graph = model.graph

    new_inits = []   # extra initializers to append
    dq_nodes = []    # DequantizeLinear / Cast nodes to prepend
    total_before, total_after = 0, 0
    n_quantized, n_kept, n_fp16 = 0, 0, 0
    skipped = []

    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        nb = len(init.raw_data)
        if nb < args.min_bytes:
            n_kept += 1
            continue
        if args.only and not any(s in init.name for s in args.only.split(",")):
            n_kept += 1
            continue
        if args.exclude and any(s in init.name for s in args.exclude.split(",")):
            n_kept += 1
            continue

        # Sensitive tensors: store FP16, Cast back to FP32 before consumers.
        if args.keep_fp16 and any(s in init.name for s in args.keep_fp16.split(",")):
            arr = onnx.numpy_helper.to_array(init)
            h = arr.astype(np.float16)
            new_inits.append(helper.make_tensor(
                init.name + "_f16", TensorProto.FLOAT16, list(h.shape),
                h.tobytes(), raw=True))
            dq_nodes.append(helper.make_node(
                "Cast", [init.name + "_f16"], [init.name],
                name=init.name + "_cast", to=TensorProto.FLOAT))
            total_before += nb
            total_after += h.nbytes
            n_fp16 += 1
            graph.initializer.remove(init)
            continue
        refs = [(node, i) for node, g in find_all_nodes(graph)
                for i, inp in enumerate(node.input) if inp == init.name]
        if not refs:
            n_kept += 1
            continue
        blocked = False
        axes = set()
        for node, i in refs:
            if node.op_type in WEIGHT_SLOTS and i in WEIGHT_SLOTS[node.op_type]:
                try:
                    axes.add(consumer_axis(node, i))
                except ValueError:
                    blocked = True
                    break
            elif node.op_type == "Gather" and i == 0:
                axes.add(0)
            elif node.op_type == "Identity" and i == 0:
                pass  # weight-sharing forward; DQ output keeps the original name
            else:
                skipped.append((init.name, f"{node.op_type}:{i}"))
                blocked = True
                break
        if blocked or not axes or len(axes) > 1:
            if not blocked:
                n_kept += 1
            continue

        arr = onnx.numpy_helper.to_array(init)
        axis = axes.pop()

        # Optional: LSTM recurrent matrix (slot 2) stays FP16 while the input
        # projection W (slot 1) quantizes - recurrent feedback amplifies
        # rounding noise, input projections do not.
        if getattr(args, "lstm_r_fp16", False) and any(
                n.op_type == "LSTM" and i == 2 for n, i in refs):
            h = arr.astype(np.float16)
            new_inits.append(helper.make_tensor(
                init.name + "_f16", TensorProto.FLOAT16, list(h.shape),
                h.tobytes(), raw=True))
            dq_nodes.append(helper.make_node(
                "Cast", [init.name + "_f16"], [init.name],
                name=init.name + "_cast", to=TensorProto.FLOAT))
            total_before += nb
            total_after += h.nbytes
            n_fp16 += 1
            graph.initializer.remove(init)
            continue

        q, scales = quantize_tensor(arr, axis, scale_search=args.scale_search)
        total_before += nb
        total_after += q.nbytes + scales.nbytes + 1
        n_quantized += 1

        base = init.name
        if len(scales) > 1:
            zp_arr = np.zeros(len(scales), dtype=np.int8)
        else:
            zp_arr = np.zeros((), dtype=np.int8)
        new_inits.extend([
            helper.make_tensor(base + "_i8", TensorProto.INT8, list(q.shape), q.tobytes(), raw=True),
            helper.make_tensor(base + "_sc", TensorProto.FLOAT, [len(scales)], scales.tobytes(), raw=True),
            helper.make_tensor(base + "_zp", TensorProto.INT8, list(zp_arr.shape), zp_arr.tobytes(), raw=True),
        ])
        dq_attrs = {"axis": axis} if len(scales) > 1 else {}
        dq_nodes.append(helper.make_node(
            "DequantizeLinear",
            inputs=[base + "_i8", base + "_sc", base + "_zp"],
            outputs=[base],
            name=base + "_dq",
            **dq_attrs,
        ))
        # Old float initializer removed; consumers read the DQ output of the same name.
        graph.initializer.remove(init)

    # Optional second pass: everything still FP32 (biases, norms, small tables)
    # becomes FP16 storage + Cast back to FP32. Same trick as the repo's fp16 file.
    if args.fp16_small:
        for init in list(graph.initializer):
            if init.data_type != TensorProto.FLOAT:
                continue
            arr = onnx.numpy_helper.to_array(init)
            h = arr.astype(np.float16)
            new_inits.append(helper.make_tensor(
                init.name + "_f16", TensorProto.FLOAT16, list(h.shape),
                h.tobytes(), raw=True))
            dq_nodes.append(helper.make_node(
                "Cast", [init.name + "_f16"], [init.name],
                name=init.name + "_cast16", to=TensorProto.FLOAT))
            total_before += len(init.raw_data)
            total_after += h.nbytes
            n_fp16 += 1
            graph.initializer.remove(init)

    for t in new_inits:
        graph.initializer.append(t)
    for i, dq in enumerate(dq_nodes):
        graph.node.insert(i, dq)

    print(f"Quantized: {n_quantized} tensors "
          f"({total_before/1024/1024:.1f} MB -> {total_after/1024/1024:.1f} MB)")
    print(f"FP16-stored: {n_fp16} tensors")
    print(f"Kept FP32: {n_kept} small tensors")
    if skipped:
        print("Skipped due to unsupported consumers:")
        for name, why in skipped[:20]:
            print(f"  {name}: {why}")

    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(f"Saved {args.dst}: {os.path.getsize(args.dst)/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
