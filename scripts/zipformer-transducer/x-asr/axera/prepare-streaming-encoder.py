#!/usr/bin/env python3
"""Turn a streaming Zipformer encoder ONNX into a static graph Pulsar2 can compile.

A streaming encoder differs from the offline one in two ways:

* it has ~115 cache tensors (attention key/value, conv, nonlin-attn state) both as
  inputs and as outputs (``new_cached_*``),
* at inference time the graph processes ``chunk_size * 2 + 13`` feature frames and
  advances by ``chunk_size * 2`` frames (960 ms for a 960 ms chunk model).

``export-onnx-streaming.py`` from icefall produces a graph with a dynamic batch
dimension and a mask that is computed from ``processed_lens`` at run time. Pulsar2
cannot handle either, so this script

1. fixes every dynamic dimension to 1,
2. freezes the ``Shape -> Gather -> Concat -> Slice`` chains,
3. constant-folds the mask chain (feed ``processed_lens`` as a constant with
   ``--freeze-input processed_lens=256``; with a fixed chunk every mask except the
   very first one is identical),
4. rewrites ``Log(1 + Exp(z))`` to ``Softplus`` (Pulsar2 has no ``Log``),
5. drops outputs that became constants (``new_processed_lens``), and
6. writes back the real output shapes (cache lengths after concatenation).

The result is numerically identical to the input graph (cosine 1.0, max|diff| < 1e-4).

Usage::

    python3 prepare-streaming-encoder.py \
        --src encoder-epoch-99-avg-1-chunk-48-left-256.onnx \
        --dst encoder-streaming-ax650.onnx \
        --freeze-input processed_lens=256 \
        --drop-output new_processed_lens
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

SHAPE_CONSUMERS = {
    "Reshape": [1],
    "Expand": [1],
    "ConstantOfShape": [0],
    "Tile": [1],
    "Slice": [1, 2, 3, 4],
    "Squeeze": [1],
    "Unsqueeze": [1],
    "ReduceMean": [1],
    "Pad": [1],
    "Split": [1],
    "TopK": [1],
}


def fix_batch(model: onnx.ModelProto) -> int:
    """把所有 dim_param 换成 1。"""
    n = 0
    def walk(vis):
        nonlocal n
        for d in vis.type.tensor_type.shape.dim:
            if d.HasField("dim_param") and d.dim_param:
                d.Clear()
                d.dim_value = 1
                n += 1
    for i in model.graph.input:
        walk(i)
    for o in model.graph.output:
        walk(o)
    for v in model.graph.value_info:
        walk(v)
    return n


DTYPES = {TensorProto.FLOAT: np.float32, TensorProto.INT64: np.int64,
          TensorProto.INT32: np.int32}


def feed_of(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    feed = {}
    for i in model.graph.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        if any(d == 0 for d in dims):
            raise SystemExit(f"input {i.name} still has a dynamic dim: {dims}")
        dtype = DTYPES.get(i.type.tensor_type.elem_type, np.float32)
        feed[i.name] = np.zeros(dims, dtype=dtype)
    return feed


def shape_inputs(model: onnx.ModelProto) -> set[str]:
    init = {i.name for i in model.graph.initializer}
    out = set()
    for n in model.graph.node:
        for idx in SHAPE_CONSUMERS.get(n.op_type, []):
            if idx < len(n.input) and n.input[idx] and n.input[idx] not in init:
                out.add(n.input[idx])
    return out


def prune(model: onnx.ModelProto) -> None:
    producers = {o: n for n in model.graph.node for o in n.output}
    keep, need, stack = set(), set(), [o.name for o in model.graph.output]
    while stack:
        name = stack.pop()
        if name in need:
            continue
        need.add(name)
        node = producers.get(name)
        if node is not None:
            keep.add(id(node))
            stack.extend(i for i in node.input if i)
    del model.graph.node[:]
    model.graph.node.extend([n for n in producers.values() if id(n) in keep])
    used = {i for n in model.graph.node for i in n.input}
    inits = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(inits)


def freeze(model: onnx.ModelProto, feed: dict) -> tuple[onnx.ModelProto, int]:
    targets = sorted(shape_inputs(model))
    if not targets:
        return model, 0
    probe = copy.deepcopy(model)
    producers = {o: n for n in probe.graph.node for o in n.output}
    for name in targets:
        node = producers.get(name)
        dtype = None
        if node is not None and node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to":
                    dtype = a.i
        probe.graph.output.append(helper.make_tensor_value_info(
            name, dtype if dtype else TensorProto.INT64, None))
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    values = sess.run(targets, feed)
    existing = {i.name for i in model.graph.initializer}
    for name, val in zip(targets, values):
        new_name = name + "__frozen"
        if new_name not in existing:
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(val), name=new_name))
            existing.add(new_name)
        for n in model.graph.node:
            n.input[:] = [new_name if i == name else i for i in n.input]
    return model, len(targets)


def const_fold(model: onnx.ModelProto, feed: dict, max_rounds: int = 4) -> int:
    """把「所有输入都是常量」的节点求值成 initializer（常量折叠）。

    流式图里 mask/位置相关的 Equal/Greater/Where/Unsqueeze 链会把 BOOL/Cast 张量
    一路喂给 Slice/Reshape，形状函数仍然失败，所以要在冻结形状之外做通用常量折叠。
    """
    total = 0
    for _ in range(max_rounds):
        init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        const_names = set(init)
        for n in model.graph.node:
            if n.op_type == "Constant":
                const_names.update(n.output)
        cand = [n for n in model.graph.node
                if n.op_type not in ("Constant",) and all(
                    (i in const_names or i == "") for i in n.input)]
        if not cand:
            break
        import onnx.shape_inference
        try:
            model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        except Exception:
            pass
        dtype_of = {v.name: v.type.tensor_type.elem_type for v in model.graph.value_info}
        dtype_of.update({o.name: o.type.tensor_type.elem_type for o in model.graph.output})
        dtype_of.update({i.name: i.type.tensor_type.elem_type for i in model.graph.input})
        probe = copy.deepcopy(model)
        prod = {o: n for n in probe.graph.node for o in n.output}
        # 只折叠「输出 dtype 已知」的节点，否则 onnxruntime 无法加载探针模型
        cand = [n for n in cand if all(o in dtype_of for o in n.output)]
        if not cand:
            print("[stream] 没有可折叠的常量节点", flush=True)
            break
        want = [o for n in cand for o in n.output]
        for name in want:
            probe.graph.output.append(
                helper.make_tensor_value_info(name, dtype_of[name], None))
        try:
            sess = ort.InferenceSession(probe.SerializeToString(),
                                        providers=["CPUExecutionProvider"])
            values = dict(zip(want, sess.run(want, feed)))
        except Exception as e:
            print(f"[stream] 常量折叠跳过（{type(e).__name__}: {str(e)[:300]}）", flush=True)
            break
        killed = set()
        existing = {i.name for i in model.graph.initializer}
        for n in cand:
            if any(o not in values for o in n.output):
                continue
            killed.add(id(n))
            for o in n.output:
                new_name = o + "__const"
                if new_name not in existing:
                    model.graph.initializer.append(
                        numpy_helper.from_array(np.asarray(values[o]), name=new_name))
                    existing.add(new_name)
                for other in model.graph.node:
                    other.input[:] = [new_name if i == o else i for i in other.input]
        if not killed:
            break
        del model.graph.node[:]
        model.graph.node.extend([n for n in prod.values() if id(n) not in killed])
        used = {i for n in model.graph.node for i in n.input}
        inits = [i for i in model.graph.initializer if i.name in used]
        del model.graph.initializer[:]
        model.graph.initializer.extend(inits)
        total += len(killed)
        print(f"[stream] 常量折叠 {len(killed)} 个节点", flush=True)
    return total


def fold_log(model: onnx.ModelProto) -> int:
    g = model.graph
    by_out = {o: n for n in g.node for o in n.output}
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    killed, patched = set(), 0

    def const(name):
        if name in init:
            return init[name]
        n = by_out.get(name)
        if n is not None and n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
        return None

    for node in g.node:
        if node.op_type != "Log":
            continue
        add = by_out.get(node.input[0])
        if add is None or add.op_type != "Add":
            continue
        exp_node, konst = None, None
        for i in add.input:
            p = by_out.get(i)
            if p is not None and p.op_type == "Exp":
                exp_node = p
            else:
                c = const(i)
                if c is not None:
                    konst = c
        if exp_node is None or konst is None:
            continue
        if np.asarray(konst).reshape(-1).size != 1 or \
                not np.isclose(float(np.asarray(konst).reshape(-1)[0]), 1.0):
            continue
        node.op_type = "Softplus"
        node.input[:] = [exp_node.input[0]]
        killed.add(id(exp_node))
        killed.add(id(add))
        patched += 1

    if killed:
        del g.node[:]
        g.node.extend([n for n in by_out.values() if id(n) not in killed])
        used = {i for n in g.node for i in n.input}
        inits = [i for i in g.initializer if i.name in used]
        del g.initializer[:]
        g.initializer.extend(inits)
    return patched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--drop-output", action="append", default=[],
                    help="输出名，从 graph.output 里去掉（例如 new_processed_lens，"
                         "它只由 frozen 的 processed_lens 推出来）")
    ap.add_argument("--freeze-input", action="append", default=[],
                    help="name=value，把某个标量输入冻结成常量（例如 processed_lens=0）")
    args = ap.parse_args()

    model = onnx.load(args.src)
    frozen_feed = {}
    for spec in args.freeze_input:
        name, val = spec.split("=")
        target = next((i for i in model.graph.input if i.name == name), None)
        if target is None:
            continue
        import numpy as _np
        dims = [d.dim_value for d in target.type.tensor_type.shape.dim]
        arr = _np.full(dims, int(val), dtype=_np.int64)
        model.graph.initializer.append(numpy_helper.from_array(arr, name=name))
        model.graph.input.remove(target)
        frozen_feed[name] = arr
        print(f"[stream] 冻结输入 {name} = {val}", flush=True)
    n = fix_batch(model)
    print(f"[stream] 固定 {n} 个动态维度 -> 1", flush=True)
    feed = feed_of(model)
    print(f"[stream] inputs: {len(feed)}", flush=True)

    total = 0
    for it in range(8):
        model, k = freeze(model, feed)
        total += k
        print(f"[stream] 第 {it+1} 轮冻结 {k} 个形状张量", flush=True)
        if k == 0:
            break
    # 输出的 cache 形状是「旧 cache + 新 chunk」拼接后的长度，导出时记录的是
    # 动态符号；这里用一次真实前向把实际形状写回 graph.output
    nfold = const_fold(model, feed)
    print(f"[stream] 常量折叠合计 {nfold} 个节点", flush=True)
    patched = fold_log(model)
    print(f"[stream] Log->Softplus: {patched}", flush=True)
    # 用实际前向修正输出形状（cache 拼接后的长度）
    probe = copy.deepcopy(model)
    for o in probe.graph.output:
        o.type.tensor_type.ClearField("shape")
    sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = sess.run(None, feed)
    for o, val in zip(model.graph.output, outs):
        arr = np.asarray(val)
        o.type.tensor_type.ClearField("shape")
        for d in arr.shape:
            o.type.tensor_type.shape.dim.add().dim_value = int(d)
    print(f"[stream] 修正 {len(outs)} 个输出形状", flush=True)

    prune(model)
    if args.drop_output:
        keep = [o for o in model.graph.output if o.name not in args.drop_output]
        n_drop = len(model.graph.output) - len(keep)
        if n_drop:
            print(f"[stream] 去掉输出 {args.drop_output}", flush=True)
            del model.graph.output[:]
            model.graph.output.extend(keep)

    # 常量折叠后有些输出（如 new_processed_lens = processed_lens + chunk）已经没有
    # 生产者节点了：Pulsar2 量化时会 KeyError，直接去掉这种输出（运行时自己维护）
    produced = {o for n in model.graph.node for o in n.output}
    dropped = [o.name for o in model.graph.output if o.name not in produced]
    if dropped:
        print(f"[stream] 去掉常量输出: {dropped}", flush=True)
        keep = [o for o in model.graph.output if o.name in produced]
        del model.graph.output[:]
        model.graph.output.extend(keep)

    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    print(f"[stream] 保存 {args.dst}: nodes={len(model.graph.node)} "
          f"inputs={len(model.graph.input)} outputs={len(model.graph.output)}", flush=True)

    ref = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"])
    new = ort.InferenceSession(args.dst, providers=["CPUExecutionProvider"])
    a = ref.run(None, {**feed, **frozen_feed})
    b = new.run(None, feed)
    cos = []
    for x, y in zip(a, b):
        x = np.asarray(x, np.float64).reshape(-1)
        y = np.asarray(y, np.float64).reshape(-1)
        cos.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)))
    print(f"[stream] cosine min={min(cos):.8f} mean={np.mean(cos):.8f} "
          f"max|diff|(out0)={np.abs(np.asarray(a[0])-np.asarray(b[0])).max():.6f}")


if __name__ == "__main__":
    main()
