#!/usr/bin/env python3
"""Prepare an exported X-ASR zipformer2 encoder for AXERA (AX650) compilation.

The encoder produced by
``icefall/egs/librispeech/ASR/zipformer/export-streaming-as-non-streaming-onnx.py``
cannot be compiled by Pulsar2 as is. Three graph rewrites are needed:

1. Fold constant ``If`` nodes.
   The exporter keeps the causal/non-causal branches of ``ConvModule`` and
   ``downsample`` as ONNX control flow (43 ``If`` nodes for X-ASR). Their
   conditions only depend on the (static) input length, so they can be
   evaluated once and inlined. Pulsar2's quantization pass fails on ``If``.

2. Freeze dynamic shape computations.
   ``NonlinAttention`` reshapes tensors with a runtime shape chain
   (``Shape -> Gather -> Concat -> Reshape``). Pulsar2 requires the target shape
   of ``Reshape`` to be a compile-time constant.

3. Rewrite the ``SwooshL/SwooshR`` activations to ``Softplus``.
   Zipformer's squashing functions are exported as the numerically stable
   ``Log(1 + Exp(-|x|))``. Pulsar2 does not support ``Log`` and splits the graph
   into 17 NPU subgraphs and 16 ONNX (CPU) subgraphs, which is both slow to
   compile and slow at runtime. Since ``log(1 + exp(z)) == softplus(z)``, each
   such pattern is replaced by a single ``Softplus`` node, which Pulsar2
   supports, giving a single NPU subgraph.

Usage::

    python3 prepare-encoder.py \
        --src encoder-epoch-99-avg-1.onnx \
        --dst encoder-ax650.onnx \
        --num-frames 1000
"""

import argparse
import copy
import logging

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def get_const(node_by_output, initializers, name):
    if name in initializers:
        return initializers[name]
    node = node_by_output.get(name)
    if node is not None and node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
    return None


DTYPES = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
}


def make_feed(model, num_frames=None):
    """Build zero inputs for every static graph input (used for graph rewrites)."""
    feed = {}
    for i in model.graph.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        if num_frames is not None and len(dims) == 3:
            dims = [1, num_frames, 80]
        dtype = DTYPES.get(i.type.tensor_type.elem_type, np.float32)
        feed[i.name] = np.zeros(dims, dtype=dtype)
    return feed


def fold_if_nodes(model):
    """Inline the branch taken by every If node whose condition is a constant."""
    graph = model.graph
    ifs = [n for n in graph.node if n.op_type == "If"]
    if not ifs:
        return model, 0

    # Evaluate all conditions at once by temporarily exposing them as outputs.
    probe = copy.deepcopy(model)
    cond_names = [n.input[0] for n in probe.graph.node if n.op_type == "If"]
    for name in cond_names:
        probe.graph.output.append(
            helper.make_tensor_value_info(name, onnx.TensorProto.BOOL, None))
    feed = make_feed(probe)
    sess = ort.InferenceSession(probe.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    values = sess.run(cond_names, feed)
    cond = {n: bool(np.asarray(v).reshape(-1)[0]) for n, v in zip(cond_names, values)}

    new_nodes = []
    for node in list(graph.node):
        if node.op_type != "If":
            new_nodes.append(node)
            continue
        branch = next(a.g for a in node.attribute
                      if a.name == ("then_branch" if cond[node.input[0]] else "else_branch"))
        prefix = (node.name or node.output[0]) + "_"
        mapped = {bi.name: node.input[i + 1] for i, bi in enumerate(branch.input)}
        inner = set(bi.name for bi in branch.input)
        inner |= {init.name for init in branch.initializer}
        inner |= {o for n in branch.node for o in n.output}

        def rename(name):
            if name in mapped:
                return mapped[name]
            if name in inner:
                return prefix + name
            return name

        for init in branch.initializer:
            new_init = copy.deepcopy(init)
            new_init.name = prefix + init.name
            graph.initializer.append(new_init)

        out_map = {prefix + bo.name: io for bo, io in zip(branch.output, node.output)}
        for n in branch.node:
            nn = copy.deepcopy(n)
            nn.name = prefix + (n.name or n.output[0])
            nn.input[:] = [rename(i) if i else i for i in nn.input]
            nn.output[:] = [out_map.get(rename(o), rename(o)) for o in nn.output]
            new_nodes.append(nn)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return model, len(ifs)


SHAPE_CONSUMERS = {
    "Reshape": [1], "Expand": [1], "ConstantOfShape": [0], "Tile": [1],
    "Slice": [1, 2, 3, 4], "Squeeze": [1], "Unsqueeze": [1], "ReduceMean": [1],
    "Pad": [1], "Split": [1], "TopK": [1],
}


def prune(model):
    graph = model.graph
    producers = {o: n for n in graph.node for o in n.output}
    keep, need, stack = set(), set(), [o.name for o in graph.output]
    while stack:
        name = stack.pop()
        if name in need:
            continue
        need.add(name)
        node = producers.get(name)
        if node is not None:
            keep.add(id(node))
            stack.extend(i for i in node.input if i)
    kept = [n for n in graph.node if id(n) in keep]
    del graph.node[:]
    graph.node.extend(kept)
    used = {i for n in graph.node for i in n.input}
    inits = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)
    return model


def freeze_shapes(model, feed):
    """Replace every non-constant shape input by a constant initializer."""
    graph = model.graph
    init = {i.name for i in graph.initializer}
    targets = set()
    for node in graph.node:
        for idx in SHAPE_CONSUMERS.get(node.op_type, []):
            if idx < len(node.input) and node.input[idx] and node.input[idx] not in init:
                targets.add(node.input[idx])
    targets = sorted(targets)
    if not targets:
        return model, 0

    probe = copy.deepcopy(model)
    for name in targets:
        probe.graph.output.append(
            helper.make_tensor_value_info(name, onnx.TensorProto.INT64, None))
    sess = ort.InferenceSession(probe.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    values = sess.run(targets, feed)

    existing = {i.name for i in model.graph.initializer}
    for name, value in zip(targets, values):
        new_name = name + "__frozen"
        if new_name not in existing:
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(value), name=new_name))
            existing.add(new_name)
        for node in model.graph.node:
            node.input[:] = [new_name if i == name else i for i in node.input]
    return model, len(targets)


def fold_swoosh(model):
    """Rewrite ``Log(1 + Exp(z))`` to ``Softplus(z)``."""
    graph = model.graph
    by_output = {o: n for n in graph.node for o in n.output}
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    killed, patched = set(), 0

    for node in graph.node:
        if node.op_type != "Log":
            continue
        add = by_output.get(node.input[0])
        if add is None or add.op_type != "Add":
            continue
        exp_node, const = None, None
        for i in add.input:
            producer = by_output.get(i)
            if producer is not None and producer.op_type == "Exp":
                exp_node = producer
            else:
                const = get_const(by_output, initializers, i)
        if exp_node is None or const is None:
            continue
        flat = np.asarray(const).reshape(-1)
        if flat.size != 1 or not np.isclose(float(flat[0]), 1.0):
            continue
        node.op_type = "Softplus"
        node.input[:] = [exp_node.input[0]]
        killed.add(id(exp_node))
        killed.add(id(add))
        patched += 1

    if killed:
        kept = [n for n in graph.node if id(n) not in killed]
        del graph.node[:]
        graph.node.extend(kept)
        used = {i for n in graph.node for i in n.input}
        inits = [i for i in graph.initializer if i.name in used]
        del graph.initializer[:]
        graph.initializer.extend(inits)
    return model, patched


def wrap_to_4d(model, num_frames):
    """Expose ``x[1, 1, T, 80]`` and reshape to ``x[1, T, 80]``.

    Pulsar2's ``tensor_layout`` enum only has NHWC/NCHW, so a 4-D input avoids
    ambiguity when the model is published.
    """
    graph = model.graph
    old_input = graph.input[0]
    shape_name = "x_4d_reshape_shape"
    graph.initializer.append(numpy_helper.from_array(
        np.array([1, num_frames, 80], dtype=np.int64), name=shape_name))
    graph.node.insert(0, helper.make_node(
        "Reshape", inputs=["x4d", shape_name], outputs=[old_input.name]))
    graph.input.remove(old_input)
    graph.input.insert(0, helper.make_tensor_value_info(
        "x4d", onnx.TensorProto.FLOAT, [1, 1, num_frames, 80]))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="ONNX model exported by icefall")
    parser.add_argument("--dst", required=True,
                        help="Path to the prepared ONNX model")
    parser.add_argument("--num-frames", type=int, default=1000,
                        help="Number of fbank frames of the static window")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = onnx.load(args.src)

    model, n = fold_if_nodes(model)
    logging.info("folded %d If nodes", n)

    feed = make_feed(model, num_frames=args.num_frames)
    total = 0
    for _ in range(6):
        model, n = freeze_shapes(model, feed)
        total += n
        if n == 0:
            break
    logging.info("froze %d shape tensors", total)
    model = prune(model)

    model, n = fold_swoosh(model)
    logging.info("rewrote %d Log nodes to Softplus", n)

    model = wrap_to_4d(model, args.num_frames)
    onnx.checker.check_model(model)
    onnx.save(model, args.dst)
    logging.info("saved %s (nodes: %d)", args.dst, len(model.graph.node))

    # Sanity check: the prepared model must reproduce the original output.
    zeros = make_feed(onnx.load(args.src), num_frames=args.num_frames)
    ref = ort.InferenceSession(args.src, providers=["CPUExecutionProvider"]).run(
        None, zeros)[0]

    probe = onnx.load(args.dst)
    prepared_input = {"x4d": np.zeros([1, 1, args.num_frames, 80], dtype=np.float32)}
    got = ort.InferenceSession(probe.SerializeToString(),
                               providers=["CPUExecutionProvider"]).run(
                                   None, prepared_input)[0]
    cos = float(ref.ravel() @ got.ravel() /
                (np.linalg.norm(ref) * np.linalg.norm(got)))
    logging.info("output shape: %s  cosine(prepared, original) = %.8f",
                 list(got.shape), cos)


if __name__ == "__main__":
    main()
