#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/flatten.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Generate a readable, explicit-wiring PyTorch module from a parsed nnet3
# network: every component becomes a named attribute, and forward() spells
# out the node graph as plain python ("flatten" the interpreter loop of
# Nnet3Torch.forward into literal code).  The generated file is checked
# row-by-row against Nnet3Torch before it is trusted.
#
# Why: Nnet3Torch evaluates the graph by walking nodes in a loop — correct
# and generic, but the wiring lives in data, not in code.  The flattened
# module makes the connections visible/editable (read it, prune layers,
# insert hooks, experiment) while staying bit-compatible.
#
# The generated module needs torch_model.build_component for the layer
# modules themselves (the per-component math is shared, not duplicated);
# what is generated is the CONNECTIONS.
#
# Usage:
#   python3 flatten.py --am-pt ./torch-model-cn/am/am.pt      # writes am_flat.py
#   python3 flatten.py --am-pt ... --out /tmp/x.py --check
#
# Library use (export_torch.py):
#   from flatten import generate_flat_module, build_flat, remap_state_dict

import argparse
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


from nnet3.components import Nnet3Spec
from nnet3.mdl_reader import (
    descriptor_names,
    inference_nodes,
    parse_descriptor,
)
from torch_model import (
    Nnet3Torch,
    component_out_dim,
    descriptor_row_refs,
    first_computable_rows,
    is_ivector_index,
    recurrent_names,
    recurrent_state_layout,
)


# ---------------------------------------------------------------------------
# recurrent groups (kaldi wires e.g. c_prev = Offset(c, -1) around
# LstmNonlinearityComponent; Nnet3Torch evaluates those row by row)
# ---------------------------------------------------------------------------


_row_refs = descriptor_row_refs  # [(name, total_offset)] a tree reads


def recurrent_clusters(spec: Nnet3Spec):
    """One entry per recurrence: (names, intra_t_order).  names are the
    mutually-recursive nodes (an SCC); intra_t_order is a topological
    order of the zero-offset dependencies between them (the order rows
    must be computed in at the same t)."""
    nodes = inference_nodes(spec.nodes)
    rec = recurrent_names(nodes)
    by_name = {n.name: n for n in nodes}

    def refs_of(name):
        n = by_name[name]
        if n.kind == "dim-range":
            return [(n.source, 0)]
        if n.kind in ("component", "output"):
            return _row_refs(parse_descriptor(n.descriptor))
        return []  # input nodes have no descriptor

    def reaches(start, targets):
        seen, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for r, _o in refs_of(x):
                if r in targets:
                    return True
                stack.append(r)
        return False

    clusters, rest = [], set(rec)
    while rest:
        start = sorted(rest)[0]
        members = {m for m in rest if reaches(m, {start}) and reaches(start, {m})}
        members.add(start)
        rest -= members
        # zero-offset edges between members: referenced member first
        order = []
        pending = set(members)
        while pending:
            ready = sorted(
                m
                for m in pending
                if not any(
                    r in pending and r != m and off == 0
                    for r, off in refs_of(m)
                    if r in members
                )
            )
            if not ready:
                raise ValueError(
                    f"flatten: zero-offset cycle in recurrent group {sorted(members)}"
                )
            order.extend(ready)
            pending -= set(ready)
        clusters.append((sorted(members), order))
    return clusters


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------


def _ident(name: str) -> str:
    """kaldi name ('tdnn1.affine') -> python identifier ('tdnn1_affine')."""
    s = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not s or s[0].isdigit():
        s = "_" + s
    return s


def var_of(node_name: str) -> str:
    return "x_" + _ident(node_name)


def attr_of(component_name: str) -> str:
    return _ident(component_name)


def _member_dim(spec: Nnet3Spec, name: str, by_name) -> int:
    n = by_name[name]
    return (
        n.dim
        if n.kind == "dim-range"
        else component_out_dim(spec.component(n.component))
    )


def tree_dim(spec: Nnet3Spec, by_name, tree) -> int:
    kind = tree[0]
    if kind == "name":
        return _member_dim(spec, tree[1], by_name)
    if kind == "append":
        return sum(tree_dim(spec, by_name, x) for x in tree[1])
    if kind in ("sum", "if_defined", "failover", "round", "offset"):
        return tree_dim(spec, by_name, tree[1])
    if kind == "switch":
        return tree_dim(spec, by_name, tree[1][0])
    if kind == "scale":
        return tree_dim(spec, by_name, tree[2])
    if kind == "const":
        return tree[2]
    if kind == "replace_index":
        return tree_dim(spec, by_name, tree[1])
    raise ValueError(f"bad descriptor tree {tree!r}")


def emit_defined(tree, t_expr: str) -> str:
    """python expr: every row the tree reads is inside [0, t_len) -- kaldi's
    IsComputable for one row (IfDefined's zero-fallback trigger)."""
    kind = tree[0]
    if kind == "name":
        return f"(0 <= ({t_expr}) < t_len)"
    if kind == "offset":
        off = tree[2]
        t2 = f"({t_expr}) + {off}" if off >= 0 else f"({t_expr}) - {-off}"
        return emit_defined(tree[1], t2)
    if kind == "replace_index":
        return emit_defined(tree[1], str(tree[3]))
    if kind in ("append", "sum"):
        return "(" + " and ".join(emit_defined(x, t_expr) for x in tree[1]) + ")"
    if kind == "scale":
        return emit_defined(tree[2], t_expr)
    if kind == "const":
        return "True"
    if kind == "round":
        return emit_defined(tree[1], f"((({t_expr}) // {tree[2]}) * {tree[2]})")
    if kind == "if_defined":
        return "True"
    raise ValueError(
        f"flatten: defined() of {kind!r} in generated code is not supported"
    )


def emit_row_expr(tree, t_expr: str, members, done, spec, by_name) -> str:
    """emit_expr for ONE row: t_expr is '_t' or a folded int literal."""
    kind = tree[0]
    if kind == "name":
        name = tree[1]
        if name not in members:
            return f"_row({var_of(name)}, {t_expr})"
        if t_expr == "_t":
            if name not in done:
                raise ValueError(f"flatten: same-t cycle at {name!r}")
            return "_r_" + _ident(name)
        d = _member_dim(spec, name, by_name)
        v = "_rl_" + _ident(name)
        return f"({v}[{t_expr}] if {t_expr} >= 0 else torch.zeros(_b, 1, {d}))"
    if kind == "offset":
        off = tree[2]
        if t_expr == "_t":
            t2 = "_t" if off == 0 else (f"_t - {-off}" if off < 0 else f"_t + {off}")
        else:
            t2 = str(int(t_expr) + off)
        return emit_row_expr(tree[1], t2, members, done, spec, by_name)
    if kind == "append":
        return (
            "torch.cat(["
            + ", ".join(
                emit_row_expr(x, t_expr, members, done, spec, by_name) for x in tree[1]
            )
            + "], dim=-1)"
        )
    if kind == "sum":
        return (
            "("
            + " + ".join(
                emit_row_expr(x, t_expr, members, done, spec, by_name) for x in tree[1]
            )
            + ")"
        )
    if kind == "scale":
        return f"({tree[1]} * {emit_row_expr(tree[2], t_expr, members, done, spec, by_name)})"
    if kind == "replace_index":
        # per-frame i-vector: read row t (see torch_model.is_ivector_index)
        t2 = t_expr if is_ivector_index(tree) else str(tree[3])
        return emit_row_expr(tree[1], t2, members, done, spec, by_name)
    if kind == "const":
        return f"torch.full((_b, 1, {tree[2]}), {tree[1]})"
    if kind == "round":
        return emit_row_expr(
            tree[1],
            f"((({t_expr}) // {tree[2]}) * {tree[2]})",
            members,
            done,
            spec,
            by_name,
        )
    if kind == "switch":
        opts = tree[1]
        alts = ", ".join(
            emit_row_expr(x, t_expr, members, done, spec, by_name) for x in opts
        )
        return f"[{alts}][({t_expr}) % {len(opts)}]"
    if kind == "if_defined":
        sub = emit_row_expr(tree[1], t_expr, members, done, spec, by_name)
        d = emit_defined(tree[1], t_expr)
        dim = tree_dim(spec, by_name, tree[1])
        return f"({sub} if {d} else torch.zeros(_b, 1, {dim}))"
    if kind == "failover":
        a = emit_row_expr(tree[1], t_expr, members, done, spec, by_name)
        b = emit_row_expr(tree[2], t_expr, members, done, spec, by_name)
        d = emit_defined(tree[1], t_expr)
        return f"({a} if {d} else {b})"
    raise ValueError(f"bad descriptor tree {tree!r}")


# ---------------------------------------------------------------------------
# descriptor -> python expression
# ---------------------------------------------------------------------------


# populated by generate_flat_module for the row-dependent forms
SPEC = None
BY_NAME = None


def emit_expr(tree, node_name_to_var) -> str:
    """One descriptor tree -> a python expression string.

    Mirrors Nnet3Torch._eval_tree exactly (same ops, same order):
      name          -> the node's variable
      Append(...)   -> torch.cat(..., dim=-1)
      Sum(...)      -> left-assoc +
      Scale(s, x)   -> s * x
      Offset(x, k)  -> _shift(x, k)          (edge-clamped time shift)
      ReplaceIndex  -> _fix_index(x, v)      (fix one time index, broadcast)
    """
    kind = tree[0]
    if kind == "name":
        return node_name_to_var(tree[1])
    if kind == "append":
        parts = [emit_expr(t, node_name_to_var) for t in tree[1]]
        return "torch.cat([" + ", ".join(parts) + "], dim=-1)"
    if kind == "sum":
        parts = [emit_expr(t, node_name_to_var) for t in tree[1]]
        return "(" + " + ".join(parts) + ")"
    if kind == "scale":
        return f"({tree[1]!r} * {emit_expr(tree[2], node_name_to_var)})"
    if kind == "offset":
        return f"_shift({emit_expr(tree[1], node_name_to_var)}, {tree[2]})"
    if kind == "replace_index":
        # ReplaceIndex(x, t, v): fix time index to v, broadcast over output
        if tree[2] != "t":
            raise ValueError(f"unsupported ReplaceIndex axis {tree[2]!r}")
        if is_ivector_index(tree):
            # per-frame i-vector rows (see torch_model.is_ivector_index)
            return emit_expr(tree[1], node_name_to_var)
        return f"_fix_index({emit_expr(tree[1], node_name_to_var)}, " f"{tree[3]})"
    if kind in ("const", "round", "switch", "if_defined", "failover"):
        # row-dependent forms: generate a per-t loop over the row emitter
        row = emit_row_expr(tree, "_t", set(), set(), SPEC, BY_NAME)
        return f"torch.cat([{row} for _t in range(t_len)], dim=1)"
    raise ValueError(f"bad descriptor tree {tree!r}")


# ---------------------------------------------------------------------------
# code generation
# ---------------------------------------------------------------------------

HEADER = '''\
# AUTO-GENERATED by flatten.py -- do not edit.
#
# Explicit-wiring PyTorch module for a parsed kaldi nnet3 network
# (source: {source}).  Every component of the node graph is a named
# attribute; forward() spells the connections out in the order kaldi
# wrote them (config lines are topologically sorted).
#
# Semantics are identical to Nnet3Torch (verified row-by-row by
# flatten.py --check):
#   * input  features [B, T, input_dim], ivector [B, T, ivector_dim]
#     (one i-vector per frame, like kaldi's online decoders)
#   * output log-probs [B, T, num_pdfs] on the full 10 ms grid
#     (3x subsampling is applied outside, see VoskAmWrapper)
#   * pure like Nnet3Torch: exactly the rows given are computed; the
#     caller supplies any receptive-field context ({left}/{right} 10 ms
#     frames; the recognizer's streams replicate edge frames -- kaldi's
#     edge rule)
#
# Layer modules come from torch_model.build_component (shared, not
# duplicated); what this file adds is the visible connection structure.

import re
from typing import List, Tuple

import torch
import torch.nn as nn

from nnet3.components import Component, Node, Nnet3Spec
from torch_model import build_component


def attr_of(name: str) -> str:
    """kaldi component name ('tdnn1.affine') -> attribute ('tdnn1_affine')."""
    s = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return ("_" + s) if (not s or s[0].isdigit()) else s


def _row(buf: torch.Tensor, t: int) -> torch.Tensor:
    """Row t of buf [B, T, D] as [B, 1, D]; out-of-range rows clamp."""
    if 0 <= t < buf.shape[1]:
        return buf[:, t : t + 1, :]
    return buf[:, :1, :] if t < 0 else buf[:, -1:, :]


def _state_rows(seq: torch.Tensor, start, n: int) -> torch.Tensor:
    """Rows [start, start + n) of seq [B, T, D] (clamped): the streaming
    state handed to the next window (see forward_streaming)."""
    idx = (torch.arange(n) + start).clamp(0, seq.shape[1] - 1)
    return seq.index_select(1, idx)


def _shift(x: torch.Tensor, k: int) -> torch.Tensor:
    """Offset(x, k): out[t] = x[t + k], clamped at the utterance edges."""
    t_len = x.shape[1]
    idx = torch.arange(k, t_len + k, device=x.device).clamp(0, t_len - 1)
    return x.index_select(1, idx)


def _fix_index(x: torch.Tensor, v: int) -> torch.Tensor:
    """ReplaceIndex(x, t, v): use time-index v of x at every output frame."""
    t_len = x.shape[1]
    t = max(0, min(v, t_len - 1))
    return x[:, t:t + 1, :].expand(-1, t_len, -1)


class FlatAm(nn.Module):
    """{source}: features + i-vector -> full-rate log-probs."""

    def __init__(self, spec: Nnet3Spec):
        super().__init__()
        self.input_dim = spec.input_dim
        self.ivector_dim = spec.ivector_dim
        self.num_pdfs = spec.output_dim
        self.left_context = spec.left_context
        self.right_context = spec.right_context
'''

FOOTER = '''
    @staticmethod
    def from_am_pt(path) -> "FlatAm":
        """Load an am.pt written by export_torch.py and return a ready net."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        a = payload["args"]
        spec = Nnet3Spec(
            components=[Component(**c) for c in a["components"]],
            nodes=[Node(**n) for n in a["nodes"]],
            input_dim=a["input_dim"], ivector_dim=a["ivector_dim"],
            output_dim=a["output_dim"],
            left_context=a["left_context"], right_context=a["right_context"])
        net = FlatAm(spec)
        net.load_state_dict(remap_state_dict(payload["state_dict"], spec))
        net.eval()
        return net


def remap_state_dict(state_dict, spec):
    """Nnet3Torch keys ('component_modules.<i>.*') -> FlatAm keys."""
    out = {}
    for k, v in state_dict.items():
        m = re.match(r"component_modules\\.(\\d+)\\.(.+)$", k)
        if not m:
            raise KeyError(f"unexpected state_dict key {k!r}")
        comp = spec.components[int(m.group(1))]
        attr = attr_of(comp.name)
        out[_OWNER.get(attr, "") + attr + "." + m.group(2)] = v
    return out
'''


def generate_flat_module(spec: Nnet3Spec, source: str = "vosk nnet3 model") -> str:
    """The full text of am_flat.py for this network."""
    lines = [
        HEADER.format(source=source, left=spec.left_context, right=spec.right_context)
    ]

    # -- __init__: one named module per component ------------------------
    # components inside a recurrent group are owned by that group's
    # scripted loop module (_Rec<i>, see recurrent_class)
    clusters = recurrent_clusters(spec)
    owner = cluster_owner(spec)
    seen = set()
    for c in spec.components:
        attr = attr_of(c.name)
        if attr in seen or hasattr(nn.Module, attr):
            raise ValueError(f"attribute name collision for component {c.name!r}")
        seen.add(attr)
        if attr in owner:
            continue
        lines.append(
            f"        self.{attr} = build_component(spec.component("
            f"{c.name!r}))  # {c.type}"
        )
    for i in range(len(clusters)):
        # TorchScript: ONNX export keeps the per-row loop as a Loop node
        # with a dynamic trip count (tracing would unroll it to the
        # dummy input's length)
        lines.append(f"        self._rec{i} = torch.jit.script(_Rec{i}(spec))")

    # streaming state of a recurrent net (same layout as Nnet3Torch)
    nodes = inference_nodes(spec.nodes)
    by_name = {n.name: n for n in nodes}
    first_row = first_computable_rows(nodes, spec.components)
    state_rows, state_names = recurrent_state_layout(nodes)
    slots, off = {}, 0
    for name in state_names:
        d = _member_dim(spec, name, by_name)
        slots[name] = (off, d)
        off += d
    state_dim = off
    lines.append(f"        self.state_rows = {state_rows}")
    lines.append(f"        self.state_dim = {state_dim}")

    # -- forward: the node graph as literal code ------------------------
    if clusters:
        lines.append(
            f"""
    def forward(self, features: torch.Tensor,
                ivector: torch.Tensor) -> torch.Tensor:
        \"\"\"[B, T, input_dim] x [B, T, ivector_dim] -> [B, T, num_pdfs]
        (one-shot: blank recurrent state).\"\"\"
        state = self.initial_state(features.shape[0])
        return self.forward_streaming(features, ivector, state, 0)[0]

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        \"\"\"Streaming state at the start of an utterance (blank rows).\"\"\"
        return torch.zeros(batch, {state_rows}, {state_dim})

    def forward_streaming(self, features: torch.Tensor, ivector: torch.Tensor,
                          state: torch.Tensor, shift):
        \"\"\"One window of a stream: (log-probs [B, T, num_pdfs], state for
        the window starting `shift` frames later).  state [B, {state_rows}, {state_dim}]
        holds rows of {", ".join(state_names)} just before each loop's
        first row (see torch_model.Nnet3Torch.forward_streaming).\"\"\"
        t_len = features.shape[1]"""
        )
    else:
        lines.append(
            """
    def forward(self, features: torch.Tensor,
                ivector: torch.Tensor) -> torch.Tensor:
        \"\"\"[B, T, input_dim] x [B, T, ivector_dim] -> [B, T, num_pdfs].\"\"\"
        t_len = features.shape[1]"""
        )
    lines.append(
        "        " + var_of("ivector") + " = ivector  # [B, T, S], one per frame"
    )
    lines.append("        " + var_of("input") + " = features")
    lines.append("")
    global SPEC, BY_NAME
    SPEC = spec
    BY_NAME = {n.name: n for n in nodes}
    trigger = {}
    for names, _order in clusters:
        for n in names:
            if n not in trigger:
                trigger[n] = names  # cluster keyed by its first member seen
    emitted = set()
    lines.append("        # node graph, in the order kaldi wrote it")

    for node in nodes:
        if node.name in emitted:
            continue
        if node.name in trigger and not any(m in emitted for m in trigger[node.name]):
            # first member of a recurrent group: call its loop module
            names = trigger[node.name]
            i = next(k for k, (ns, _o) in enumerate(clusters) if ns == names)
            order = clusters[i][1]
            ext = cluster_inputs(names, by_name)
            lines.append("")
            lines.append(f"        # recurrent group ({', '.join(names)})")
            lines.append(
                "        "
                + ", ".join(var_of(m) for m in order)
                + f", = self._rec{i}("
                + ", ".join([var_of(e) for e in ext] + ["state"])
                + ")"
            )
            emitted.update(names)
            lines.append("")
            continue
        v = var_of(node.name)
        if node.kind == "input":
            continue  # prologue above
        if node.kind == "dim-range":
            src = var_of(node.source)
            lines.append(
                f"        {v} = {src}[:, :, "
                f"{node.dim_offset}:{node.dim_offset + node.dim}]"
            )
        elif node.kind == "component":
            expr = emit_expr(parse_descriptor(node.descriptor), var_of)
            attr = attr_of(node.component)
            comment = (
                ""
                if expr == var_of(node.descriptor.strip())
                else f"   # {node.descriptor}"
            )
            lines.append(f"        {v} = self.{attr}({expr}){comment}")
        elif node.kind == "output":
            expr = emit_expr(parse_descriptor(node.descriptor), var_of)
            lines.append(f"        out = {expr}   # {node.descriptor}")
            if not clusters:
                lines.append("        return out")
                continue
            lines.append("        # state for the window starting `shift` frames later")
            lines.append("        state_out = torch.cat([")
            for name in state_names:
                lines.append(
                    f"            _state_rows({var_of(name)}, "
                    f"{first_row[name] - state_rows} + shift, {state_rows}),"
                )
            lines.append("        ], dim=-1)")
            lines.append("        return out, state_out")
        else:
            raise ValueError(f"bad node kind {node.kind!r}")

    lines.append(FOOTER)
    lines.append(f"_OWNER = {owner!r}")
    for i, (names, order) in enumerate(clusters):
        lines.append("")
        lines.append(
            recurrent_class(
                i, names, order, spec, by_name, first_row, slots, state_rows
            )
        )
    return "\n".join(lines) + "\n"


def cluster_owner(spec: Nnet3Spec) -> dict:
    """component attribute -> state_dict prefix ('_rec<i>.') for the
    components that live inside a recurrent group's loop module."""
    by_name = {n.name: n for n in inference_nodes(spec.nodes)}
    out = {}
    for i, (names, _order) in enumerate(recurrent_clusters(spec)):
        for m in names:
            if by_name[m].kind == "component":
                out[attr_of(by_name[m].component)] = f"_rec{i}."
    return out


def cluster_inputs(names, by_name) -> list:
    """Non-member nodes a recurrent group reads (the loop's inputs)."""
    members = set(names)
    ext = []
    for m in names:
        n = by_name[m]
        refs = (
            [n.source]
            if n.kind == "dim-range"
            else [r for r, _o in _row_refs(parse_descriptor(n.descriptor))]
        )
        for r in refs:
            if r not in members and r not in ext:
                ext.append(r)
    return sorted(ext)


def recurrent_class(
    i, names, order, spec, by_name, first_row, slots, state_rows
) -> str:
    """Source of _Rec<i>: one recurrent group evaluated row by row.  Rows
    below a member's first computable row are blank, except the
    `state_rows` rows just before it of a fed-back member (`slots`: name
    -> (offset, dim) in the state), which come from the streaming state
    -- as in Nnet3Torch._eval_row.  TorchScript-compatible, so ONNX
    export emits a Loop with a dynamic trip count."""
    members = set(names)
    ext = cluster_inputs(names, by_name)
    comps = [by_name[m].component for m in order if by_name[m].kind == "component"]
    out = [f"class _Rec{i}(nn.Module):"]
    out.append(f'    """recurrent group: {", ".join(names)}"""')
    out.append("")
    out.append("    def __init__(self, spec: Nnet3Spec):")
    out.append("        super().__init__()")
    for c in comps:
        out.append(
            f"        self.{attr_of(c)} = build_component(spec.component({c!r}))"
        )
    out.append("")
    args = ", ".join(f"{var_of(e)}: torch.Tensor" for e in ext)
    ret = ", ".join("torch.Tensor" for _ in order)
    out.append(f"    def forward(self, {args}, state: torch.Tensor) -> Tuple[{ret}]:")
    out.append(f"        _b = {var_of(ext[0])}.shape[0]")
    out.append(f"        t_len = {var_of(ext[0])}.shape[1]")
    for m in order:
        out.append(f"        _rl_{_ident(m)}: List[torch.Tensor] = []")
    out.append("        for _t in range(t_len):")
    done = set()
    for m in order:
        mn = by_name[m]
        rv = "_r_" + _ident(m)
        if mn.kind == "dim-range":
            out.append(
                f"            {rv} = _r_{_ident(mn.source)}"
                f"[:, :, {mn.dim_offset}:{mn.dim_offset + mn.dim}]"
            )
        else:
            expr = emit_row_expr(
                parse_descriptor(mn.descriptor), "_t", members, done, spec, by_name
            )
            out.append(f"            {rv} = self.{attr_of(mn.component)}({expr})")
        f = first_row[m]
        if m in slots:
            o, d = slots[m]
            k = f"_t - {f - state_rows}"
            out.append(f"            if _t < {f - state_rows}:")
            out.append(f"                {rv} = torch.zeros_like({rv})")
            out.append(f"            elif _t < {f}:  # carried streaming state")
            out.append(f"                {rv} = state[:, {k}:{k} + 1, {o}:{o + d}]")
        elif f > 0:
            out.append(f"            if _t < {f}:")
            out.append(f"                {rv} = torch.zeros_like({rv})")
        out.append(f"            _rl_{_ident(m)}.append({rv})")
        done.add(m)
    out.append(
        "        return ("
        + ", ".join(f"torch.cat(_rl_{_ident(m)}, dim=1)" for m in order)
        + ",)"
    )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------


def remap_state_dict(state_dict, spec):
    """Nnet3Torch keys ('component_modules.<i>.*') -> FlatAm keys.

    Also defined in the generated file (so am_flat.py is self-contained);
    this copy is what export_torch.py / flatten.py --check use.
    """
    owner = cluster_owner(spec)
    out = {}
    for k, v in state_dict.items():
        m = re.match(r"component_modules\.(\d+)\.(.+)$", k)
        if not m:
            raise KeyError(f"unexpected state_dict key {k!r}")
        comp = spec.components[int(m.group(1))]
        attr = attr_of(comp.name)
        out[owner.get(attr, "") + attr + "." + m.group(2)] = v
    return out


def build_flat(spec: Nnet3Spec, source: str = "vosk nnet3 model"):
    """Generate, import and instantiate the flat module for ``spec``.

    The code goes through a real file (not exec of a string):
    torch.jit.script, used for the recurrent loop modules, reads the
    class source with inspect.  The file is deleted once imported.
    """
    code = generate_flat_module(spec, source=source)
    fd, path = tempfile.mkstemp(prefix="am_flat_", suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)
        flat_cls = load_flat_class(Path(path))
        return flat_cls(spec)  # scripting happens here, source still on disk
    finally:
        os.remove(path)


def check_parity(
    ref: torch.nn.Module, flat: torch.nn.Module, feats: torch.Tensor, iv: torch.Tensor
) -> float:
    """Max abs difference between Nnet3Torch and FlatAm on one input."""
    ref.eval(), flat.eval()
    with torch.no_grad():
        a = ref(feats, iv)
        b = flat(feats, iv)
    assert a.shape == b.shape, (a.shape, b.shape)
    return (a - b).abs().max().item()


def load_flat_class(code_path: Path):
    """Import a generated am_flat.py (for external verification)."""
    spec = importlib.util.spec_from_file_location("am_flat", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FlatAm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args():
    p = argparse.ArgumentParser(
        description="Generate am_flat.py (explicit-wiring PyTorch module) "
        "from an am.pt written by export_torch.py."
    )
    p.add_argument("--am-pt", required=True, help="path to am/am.pt")
    p.add_argument(
        "--out", default=None, help="output .py (default: am_flat.py next to am.pt)"
    )
    p.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="verify against Nnet3Torch at several shapes " "(default on)",
    )
    p.add_argument("--no-check", dest="check", action="store_false")
    return p.parse_args()


def main():
    args = get_args()
    am_pt = Path(args.am_pt)
    out = Path(args.out) if args.out else am_pt.parent / "am_flat.py"

    payload = torch.load(am_pt, map_location="cpu", weights_only=False)
    ref = Nnet3Torch.from_export_args(payload["args"])
    ref.load_state_dict(payload["state_dict"])
    ref.eval()
    spec = ref.spec

    code = generate_flat_module(spec, source=am_pt.parent.parent.name)
    out.write_text(code)
    print(f"wrote {out}  ({len(code.splitlines())} lines)")

    if args.check:
        flat = build_flat(spec, source=am_pt.parent.parent.name)
        flat.load_state_dict(remap_state_dict(payload["state_dict"], spec))
        rng = torch.Generator().manual_seed(0)
        worst = 0.0
        for t in (1, 7, 40, 137):
            feats = torch.randn(2, t, spec.input_dim, generator=rng)
            iv = 0.1 * torch.randn(2, t, spec.ivector_dim, generator=rng)
            d = check_parity(ref, flat, feats, iv)
            worst = max(worst, d)
            print(f"  T={t:4d}  max|Nnet3Torch - FlatAm| = {d:.2e}")
        ok = worst < 1e-6
        print("PARITY", "OK" if ok else f"FAILED ({worst:.2e})")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
