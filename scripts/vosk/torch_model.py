# sherpa-onnx/scripts/vosk/torch_model.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# PyTorch evaluation of a parsed kaldi nnet3 network (Nnet3Spec from
# nnet3/mdl_reader.py).  This is the middle stage of the export pipeline:
#
#     final.mdl --(nnet3/mdl_reader)--> Nnet3Spec --(this file)--> PyTorch
#     --(export_onnx.py)--> final.onnx
#
# Design notes (details in nnet3/notes.md):
#
#   * Every component becomes a small module whose forward pass is the
#     component's *inference-time* math (training-only behavior is dropped).
#   * The node graph (component-node / dim-range-node descriptors) is
#     evaluated in the order kaldi wrote it (config lines are topologically
#     sorted), over tensors shaped [batch, time, dim].
#   * Time semantics: all nodes preserve the time axis; TdnnComponent looks
#     at neighbouring frames (its TimeOffsets) with *edge clamping* — the
#     same convention kaldi's decodables use when they assemble chunks
#     (decodable-online-looped.cc clamps input_frame to [0, T-1]).
#   * forward() computes the network at EVERY input frame (full 10 ms grid).
#     The frame subsampling (kaldi anchors decoder frame k at input frame
#     k*factor -- nnet-compile-looped.cc:143-146, the grid always
#     starts at input frame 0) is applied by VoskAmWrapper keeping rows
#     ::factor; the full-rate output enables
#     the row-by-row regression check R[k] == O[k*factor].  For the export,
#     DecimationPlan computes those rows directly (kaldi-style row
#     sparsity): the decimated zone runs on the row lattice, identical
#     values at ~1/factor the rows.

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# nnet3/ is imported as a plain subpackage (scripts are run with scripts/vosk
# as the working directory / on sys.path; see run.sh and the notes).
from nnet3.components import IDENTITY_TYPES, Component, Node, Nnet3Spec
from nnet3.mdl_reader import descriptor_names, inference_nodes, parse_descriptor


# ---------------------------------------------------------------------------
# Per-component modules (inference-time math only)
# ---------------------------------------------------------------------------


class FixedAffine(nn.Module):
    """y = x @ W^T + b with a constant W (kaldi FixedAffineComponent).

    Used e.g. for the idct layer at the network's input.
    """

    def __init__(self, c: Component):
        super().__init__()
        w = c.params["LinearParams"]  # [out, in]
        self.weight = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(w)), requires_grad=False
        )
        b = c.params["BiasParams"]  # [out] (may be empty)
        if b.size:
            self.bias = nn.Parameter(
                torch.from_numpy(np.ascontiguousarray(b)), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class BatchNormTestMode(nn.Module):
    """kaldi BatchNormComponent in test mode — a frozen affine.

    From kaldi's ComputeDerived() (nnet-normalize-component.cc):
        mean  = StatsMean
        var   = StatsVar           (floored at 0)
        scale = (var + eps)^(-0.5) * TargetRms
        offset= -mean * scale
        y     = x * scale + offset
    (kaldi sets test mode at load time; vosk's model.cc does the same.)

    If BlockDim < Dim the parameters repeat per block (kaldi normalizes each
    block separately); the per-block vectors are tiled to full Dim here.
    """

    def __init__(self, c: Component):
        super().__init__()
        dim = int(c.params["Dim"])
        block = int(c.params.get("BlockDim", dim))
        eps = float(c.params["Epsilon"])
        target_rms = float(c.params.get("TargetRms", 1.0))
        mean = c.params["StatsMean"].astype(np.float64)
        var = np.maximum(c.params["StatsVar"].astype(np.float64), 0.0)
        scale = (var + eps) ** -0.5 * target_rms
        offset = -mean * scale
        if block != dim:  # tile per-block parameters
            reps = dim // block
            scale = np.tile(scale, reps)
            offset = np.tile(offset, reps)
        self.scale = nn.Parameter(
            torch.from_numpy(scale.astype(np.float32)), requires_grad=False
        )
        self.offset = nn.Parameter(
            torch.from_numpy(offset.astype(np.float32)), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.offset


class Tdnn(nn.Module):
    """kaldi TdnnComponent: y(t) = b + sum_i W_i @ x(t + offset_i).

    LinearParams layout is [out_dim, in_dim * num_offsets]; column block i
    belongs to TimeOffsets[i] (nnet-convolutional-component.h).  Frames
    outside the utterance are clamped to the edge (kaldi's chunk assembly
    does the same).
    """

    def __init__(self, c: Component):
        super().__init__()
        self.offsets: List[int] = c.params["TimeOffsets"]
        w = c.params["LinearParams"]  # [out, in * n_offsets]
        out_dim, cols = w.shape
        self.in_dim = cols // len(self.offsets)
        # one weight block per offset, stored as [n_offsets, out, in]
        blocks = [
            w[:, i * self.in_dim : (i + 1) * self.in_dim]
            for i in range(len(self.offsets))
        ]
        self.weight = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(np.stack(blocks))),
            requires_grad=False,
        )
        b = c.params["BiasParams"]
        self.bias = (
            nn.Parameter(torch.from_numpy(np.ascontiguousarray(b)), requires_grad=False)
            if b.size
            else None
        )

    def forward(self, x: torch.Tensor, row_step: int = 1) -> torch.Tensor:
        # x: [B, T, in].  row_step > 1 evaluates on a decimated row grid
        # (row j = full-rate row j*row_step), where an offset o
        # becomes o/row_step steps; DecimationPlan guarantees divisibility.
        t_len = x.shape[1]
        out = None
        for i, off in enumerate(self.offsets):
            o = off // row_step
            assert o * row_step == off, (off, row_step)
            # clamped gather of x(t + o); same edge policy as kaldi
            idx = torch.arange(o, t_len + o, device=x.device)
            idx = idx.clamp(0, t_len - 1)
            shifted = x.index_select(1, idx)  # [B, T, in]
            y = F.linear(shifted, self.weight[i])  # [B, T, out]
            out = y if out is None else out + y
        if self.bias is not None:
            out = out + self.bias
        return out


class Affine(nn.Module):
    """y = x @ W^T + b — NaturalGradientAffineComponent (and the bias-less
    LinearComponent); the natural-gradient machinery is training-only."""

    def __init__(self, c: Component):
        super().__init__()
        w = c.params.get("LinearParams", c.params.get("Params"))
        self.weight = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(w)), requires_grad=False
        )
        b = c.params.get("BiasParams")
        if b is not None and b.size:
            self.bias = nn.Parameter(
                torch.from_numpy(np.ascontiguousarray(b)), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Permute(nn.Module):
    """kaldi PermuteComponent: out[:, i] = in[:, ColumnMap[i]]
    (nnet-simple-component.cc: Propagate = CopyCols(in, column_map_))."""

    def __init__(self, c: Component):
        super().__init__()
        self.register_buffer(
            "colmap",
            torch.tensor([int(x) for x in c.params["ColumnMap"]], dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, self.colmap)


class TimeHeightConvolution(nn.Module):
    """kaldi TimeHeightConvolutionComponent: 2-D convolution over
    (time x height).

    Verified semantics (kaldi nnet3/convolution-test.cc
    ConvolveForwardSimple, the reference implementation):

      * input  [B, T, height_in * num_filters_in], layout [height, filter];
      * output [B, T, height_out * num_filters_out], same layout;
      * LinearParams [num_filters_out, num_offsets * num_filters_in],
        column block o belongs to Offsets[o] = (time_offset, height_offset);
      * BiasParams [num_filters_out], broadcast over time and height;
      * for each offset: input row t+dt (ZERO-PAD at time edges -- unlike
        Tdnn's clamping), input height h_in = h_out*subsample + dh
        (skipped when outside [0, height_in) -- zero contribution).

    Implemented as patch-gemm: one gather+concat, one matmul per layer.
    """

    def __init__(self, c: Component):
        super().__init__()
        self.f_in = int(c.params["NumFiltersIn"])
        self.f_out = int(c.params["NumFiltersOut"])
        self.height_in = int(c.params["HeightIn"])
        self.height_out = int(c.params["HeightOut"])
        self.subsample = int(c.params["HeightSubsampleOut"])
        self.offsets = [(int(t), int(h)) for t, h in c.params["Offsets"]]
        w = c.params["LinearParams"]  # [f_out, n_off * f_in]
        b = c.params["BiasParams"]  # [f_out]
        self.weight = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(w)), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(b)), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, height_in * f_in] -> [B, T, height_out * f_out]
        B, T, _ = x.shape
        x4 = x.reshape(B, T, self.height_in, self.f_in)

        # time-shifted copies with zero padding at the edges
        idx = torch.arange(T, device=x.device)
        parts = []
        for dt, dh in self.offsets:
            src = idx + dt
            valid_t = ((src >= 0) & (src < T)).to(x.dtype)
            xt = x4.index_select(1, src.clamp(0, T - 1)) * valid_t.view(1, T, 1, 1)
            # height rows with zero padding at the edges
            h_idx = torch.arange(self.height_out, device=x.device) * self.subsample + dh
            valid_h = ((h_idx >= 0) & (h_idx < self.height_in)).to(x.dtype)
            xh = xt.index_select(2, h_idx.clamp(0, self.height_in - 1)) * valid_h.view(
                1, 1, -1, 1
            )
            parts.append(xh.reshape(B, T, self.height_out, self.f_in))
        patch = torch.cat(parts, dim=-1)  # [B, T, h_out, n_off*f_in]
        out = (
            patch.reshape(-1, self.f_in * len(self.offsets)) @ self.weight.t()
            + self.bias
        )
        return out.reshape(B, T, self.height_out * self.f_out)


class LogSoftmax(nn.Module):
    """kaldi LogSoftmaxComponent: log-softmax over the feature dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(x, dim=-1)


class Identity(nn.Module):
    """NoOp / GeneralDropout / SpecAugment: identity at inference."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_component(c: Component) -> nn.Module:
    """Instantiate the inference module for one parsed component."""
    if c.type in IDENTITY_TYPES:
        return Identity()
    if c.type == "FixedAffineComponent":
        return FixedAffine(c)
    if c.type == "BatchNormComponent":
        return BatchNormTestMode(c)
    if c.type == "TdnnComponent":
        return Tdnn(c)
    if c.type in ("NaturalGradientAffineComponent", "LinearComponent"):
        return Affine(c)
    if c.type == "PermuteComponent":
        return Permute(c)
    if c.type == "TimeHeightConvolutionComponent":
        return TimeHeightConvolution(c)
    if c.type == "RectifiedLinearComponent":
        return nn.ReLU()
    if c.type == "LogSoftmaxComponent":
        return LogSoftmax()
    raise ValueError(f"no torch module for component type {c.type!r} ({c.name})")


# ---------------------------------------------------------------------------
# Decimation plan -- compute only the rows the decoder reads
# ---------------------------------------------------------------------------


def _descriptor_time_deps(tree):
    """(offsets, unknown) for a descriptor tree: how the expression's output
    rows depend on the rows of the names it references.  ``unknown`` marks
    constructs whose dependency is not a pure time shift (ReplaceIndex pins
    one row) -- those force full-rate bookkeeping."""
    kind = tree[0]
    if kind == "name":
        return {0}, False
    if kind in ("append", "sum"):
        offs, unk = set(), False
        for t in tree[1]:
            o, u = _descriptor_time_deps(t)
            offs |= o
            unk = unk or u
        return offs, unk
    if kind == "scale":
        return _descriptor_time_deps(tree[2])
    if kind == "offset":
        o, u = _descriptor_time_deps(tree[1])
        return {x + tree[2] for x in o}, u
    return {0}, True  # replace_index or unknown: conservative


def _component_time_deps(c: Component):
    """Time offsets the component's output rows have on its input rows
    (None = unknown semantics)."""
    if c.type == "TdnnComponent":
        return {int(o) for o in c.params["TimeOffsets"]}
    if c.type == "TimeHeightConvolutionComponent":
        return {int(t) for t, _h in c.params["Offsets"]}
    return {0}  # per-row components (affine, batchnorm, relu, ...)


def _component_mac(c: Component) -> float:
    """Weight matmul MACs per output row (elementwise ops count as 0)."""
    p = c.params
    if c.type in (
        "FixedAffineComponent",
        "NaturalGradientAffineComponent",
        "LinearComponent",
        "TdnnComponent",
        "TimeHeightConvolutionComponent",
    ):
        w = p.get("LinearParams", p.get("Params"))
        return float(w.shape[0] * w.shape[1])
    return 0.0


class DecimationPlan:
    """Row-selection plan: kaldi's row-level sparsity over the output rows.

    Kaldi's computation-graph builder computes, per node, only the rows the
    requested output rows depend on (computation-graph.cc); the looped
    request asks only for rows ``t += frame_subsampling_factor`` (no
    offset) -- nnet-compile-looped.cc -- i.e. decoder row k is network
    row ``k*factor``.  This class reproduces that row selection:

    * each node's needed rows are tracked as residue classes modulo
      ``factor`` (time offsets shift the residues);
    * a node is *decimatable* if its time offsets are all multiples of
      ``factor`` -- its rows of one residue class then form a closed lattice
      it can be evaluated on directly (offsets rescaled by ``factor``);
    * the decimated zone is the maximal consumer-closed decimatable set
      reaching the output (a node consumed by a full-rate node must itself
      be full-rate).  Everything else is computed full-rate and sliced at
      the boundary -- identical values to computing everything and slicing
      the output, at ~1/factor the rows inside the decimated zone.
    """

    def __init__(self, spec: Nnet3Spec, factor: int):
        assert factor > 1
        self.factor = factor
        comps = {c.name: c for c in spec.components}

        # only the nodes forward() evaluates (the xent branch is not)
        nodes = inference_nodes(spec.nodes)
        by_name = {n.name: n for n in nodes}

        # -- per-node: dependency offsets / referenced names --------------
        offsets: Dict[str, "set|None"] = {}  # output rows -> referenced rows
        refs: Dict[str, set] = {}
        for n in nodes:
            desc = parse_descriptor(n.descriptor) if n.descriptor else None
            doffs, dunknown = (
                _descriptor_time_deps(desc) if desc is not None else ({0}, False)
            )
            names = descriptor_names(desc) if desc is not None else set()
            if n.kind == "dim-range" and n.source:
                names = names | {n.source}
            refs[n.name] = names
            if dunknown:
                offsets[n.name] = None
                continue
            if n.kind == "component":
                co = _component_time_deps(comps[n.component])
                offsets[n.name] = {c + d for c in co for d in doffs}
            else:  # input / dim-range / output: per-row plus descriptor shifts
                offsets[n.name] = set(doffs)

        # -- consumers (within the evaluated prefix) ---------------------
        consumers: Dict[str, set] = {n.name: set() for n in nodes}
        for n in nodes:
            for r in refs[n.name]:
                if r in consumers:
                    consumers[r].add(n.name)

        # -- needed rows, as residue classes modulo `factor` -------------
        # The row grid starts at residue 0 (row k = frame k*factor); a
        # consumer reading rows of its own residue set `S` through time
        # offsets `O` needs its input at residues S + O (mod factor).
        out_name = nodes[-1].name  # the evaluated output node
        all_residues = set(range(factor))

        def shift(residues, offset):
            return {(r + offset) % factor for r in residues}

        needed: Dict[str, set] = {out_name: {0}}
        for n in reversed(nodes[:-1]):
            residues = set()
            for consumer in consumers[n.name]:
                consumed = needed[consumer]
                if offsets[consumer] is None:  # unknown dependency
                    residues |= all_residues
                else:
                    for offset in offsets[consumer]:
                        residues |= shift(consumed, offset)
            # a dead node (no consumers) is never read
            needed[n.name] = residues if consumers[n.name] else all_residues

        # -- decimated zone: decimatable + consumer-closed ---------------
        def decimatable(n) -> bool:
            if n.kind not in ("component", "dim-range"):
                return False
            if n.kind == "component" and comps[n.component].type == (
                "TimeHeightConvolutionComponent"
            ):
                return False  # zero-pad edges: keep full-rate (conservative)
            o = offsets[n.name]
            return o is not None and all(x % factor == 0 for x in o)

        top = {out_name}
        for n in reversed(nodes[:-1]):
            if (
                n.name not in top
                and decimatable(n)
                and consumers[n.name]
                and all(c in top for c in consumers[n.name])
            ):
                top.add(n.name)

        self.top = top
        self.nodes = [n.name for n in nodes]
        self.zones = {
            n.name: (
                "input"
                if n.kind == "input"
                else "decimated"
                if n.name in top
                else "transition"
                if len(needed[n.name]) < factor
                else "dense"
            )
            for n in nodes
        }
        self.needed_residues = needed
        self.mac = {
            n.name: _component_mac(comps[n.component]) if n.kind == "component" else 0.0
            for n in nodes
        }
        # full-rate values the decimated zone slices at the boundary
        self.boundary = sorted(
            {
                r
                for n in nodes
                if n.name in top
                for r in refs[n.name]
                if r not in top and by_name[r].kind != "input"
            }
        )
        total = sum(self.mac.values())
        used = sum(v for k, v in self.mac.items() if self.zones[k] != "decimated")
        used += sum(v for k, v in self.mac.items() if self.zones[k] == "decimated") / (
            factor
        )
        self.compute_ratio = used / total if total else 1.0
        # what exact per-row masks (compute only the rows a node really needs)
        # could additionally save:
        extra = sum(
            v * (1 - len(needed[k]) / factor)
            for k, v in self.mac.items()
            if self.zones[k] == "transition"
        )
        self.transition_extra_ratio = extra / total if total else 0.0

    def needed_rows(self, name: str) -> str:
        """e.g. '1/3' -- the share of rows the node's output needs."""
        return f"{len(self.needed_residues[name])}/{self.factor}"


# ---------------------------------------------------------------------------
# The whole network
# ---------------------------------------------------------------------------


class Nnet3Torch(nn.Module):
    """Evaluates a parsed Nnet3Spec on [B, T, D] tensors.

    forward(features, ivector) returns the network output for EVERY input
    frame: [B, T, num_pdfs] on the full 10 ms grid (see the module docstring
    for why; VoskAmWrapper applies the 3x subsampling afterwards).
    """

    def __init__(self, spec: Nnet3Spec):
        super().__init__()
        self.spec = spec
        # One module per component (identity modules included, so node lookup
        # is uniform).  A ModuleList + name map, because kaldi component names
        # contain dots ("tdnn1.affine") which nn.ModuleDict keys forbid.
        self.component_modules = nn.ModuleList(
            [build_component(c) for c in spec.components]
        )
        self._name_to_idx = {c.name: i for i, c in enumerate(spec.components)}
        # the nodes the "output" node depends on, ending with it (training
        # branches such as output-xent are left out; see inference_nodes)
        self.nodes: List[Node] = inference_nodes(spec.nodes)
        self._plans: Dict[tuple, DecimationPlan] = {}

    # -- descriptor evaluation -------------------------------------------

    def _eval_descriptor(
        self,
        desc: str,
        nodes_out: Dict[str, torch.Tensor],
        batch: int,
        t_len: int,
        features: torch.Tensor,
        ivector: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one input= descriptor expression to a [B, T, D] tensor."""
        tree = parse_descriptor(desc)
        return self._eval_tree(tree, nodes_out, batch, t_len, features, ivector)

    def _eval_tree(self, tree, nodes_out, batch, t_len, features, ivector):
        kind = tree[0]
        if kind == "name":
            return nodes_out[tree[1]]
        if kind == "append":
            parts = [
                self._eval_tree(t, nodes_out, batch, t_len, features, ivector)
                for t in tree[1]
            ]
            return torch.cat(parts, dim=-1)
        if kind == "sum":
            parts = [
                self._eval_tree(t, nodes_out, batch, t_len, features, ivector)
                for t in tree[1]
            ]
            out = parts[0]
            for p in parts[1:]:
                out = out + p
            return out
        if kind == "scale":
            return tree[1] * self._eval_tree(
                tree[2], nodes_out, batch, t_len, features, ivector
            )
        if kind == "offset":
            # time shift: out[t] = x[t + off], clamped to the utterance edges
            x = self._eval_tree(tree[1], nodes_out, batch, t_len, features, ivector)
            idx = torch.arange(tree[2], t_len + tree[2], device=x.device)
            idx = idx.clamp(0, t_len - 1)
            return x.index_select(1, idx)
        if kind == "replace_index":
            # ReplaceIndex(x, t, v): use time-index v of x at every output
            # frame (kaldi fixes one index; used for the i-vector).
            x = self._eval_tree(tree[1], nodes_out, batch, t_len, features, ivector)
            _, _expr, index_name, value = tree
            if index_name != "t":
                raise ValueError(f"unsupported ReplaceIndex axis {index_name!r}")
            t = max(0, min(value, x.shape[1] - 1))
            return x[:, t : t + 1, :].expand(-1, t_len, -1)
        raise ValueError(f"bad descriptor tree {tree!r}")

    def _eval_tree_lat(self, tree, lattice, factor: int):
        """_eval_tree on the decimated row grid: 'name' leaves resolve to
        lattice values (full-rate values are sliced on first use) and time
        offsets are rescaled by ``factor``.  ReplaceIndex etc. are excluded
        by the DecimationPlan."""
        kind = tree[0]
        if kind == "name":
            return lattice(tree[1])
        if kind == "append":
            return torch.cat(
                [self._eval_tree_lat(t, lattice, factor) for t in tree[1]], dim=-1
            )
        if kind == "sum":
            parts = [self._eval_tree_lat(t, lattice, factor) for t in tree[1]]
            out = parts[0]
            for p in parts[1:]:
                out = out + p
            return out
        if kind == "scale":
            return tree[1] * self._eval_tree_lat(tree[2], lattice, factor)
        if kind == "offset":
            x = self._eval_tree_lat(tree[1], lattice, factor)
            assert tree[2] % factor == 0, (tree[2], factor)
            o = tree[2] // factor
            t_len = x.shape[1]
            idx = torch.arange(o, t_len + o, device=x.device).clamp(0, t_len - 1)
            return x.index_select(1, idx)
        raise ValueError(f"decimated zone: unsupported descriptor {tree!r}")

    # -- forward ---------------------------------------------------------

    def forward(self, features: torch.Tensor, ivector: torch.Tensor) -> torch.Tensor:
        """
        :param features: [B, T, input_dim] float32 (e.g. 40 MFCC)
        :param ivector:  [B, ivector_dim] float32 (constant over the
                         utterance; broadcast in ReplaceIndex)
        :return: [B, T, num_pdfs] log-probs, one row per input frame

        The net is PURE: it computes exactly the rows it is given.  Edge
        semantics are the caller's job -- to match kaldi at the edges of
        an utterance, feed replicated edge frames beyond the real input
        (kaldi clamps out-of-range input indexes to the edge; the
        recognizer's streams do this in get_frames).
        """
        batch, t_len = features.shape[0], features.shape[1]
        iv = ivector[:, None, :].expand(-1, t_len, -1)  # [B, T, S]

        nodes_out: Dict[str, torch.Tensor] = {}
        for node in self.nodes:
            if node.kind == "input":
                if node.name == "input":
                    nodes_out[node.name] = features
                elif node.name == "ivector":
                    nodes_out[node.name] = iv
                else:
                    raise ValueError(f"unknown input node {node.name!r}")
            elif node.kind == "dim-range":
                src = nodes_out[node.source]
                nodes_out[node.name] = src[
                    :, :, node.dim_offset : node.dim_offset + node.dim
                ]
            elif node.kind == "component":
                x = self._eval_descriptor(
                    node.descriptor, nodes_out, batch, t_len, features, ivector
                )
                nodes_out[node.name] = self.component_modules[
                    self._name_to_idx[node.component]
                ](x)
            elif node.kind == "output":
                return self._eval_descriptor(
                    node.descriptor, nodes_out, batch, t_len, features, ivector
                )
        raise ValueError("no output node evaluated")

    # -- decimated forward (output rows only) ----------------------------

    def decimation_plan(self, factor: int) -> DecimationPlan:
        if factor not in self._plans:
            self._plans[factor] = DecimationPlan(self.spec, factor)
        return self._plans[factor]

    def forward_lattice(
        self, features: torch.Tensor, ivector: torch.Tensor, factor: int
    ) -> torch.Tensor:
        """:return: [B, ceil(T/factor), num_pdfs] -- only the rows
        the decoder reads (network rows ::factor), like kaldi's
        sparse computation request.

        Pure like forward(): exactly the rows given are computed.  The
        decimated zone runs on the row lattice ``::factor`` of the input
        (time offsets rescaled by factor); values equal
        ``forward(...)[:, ::factor, :]`` exactly -- the same expressions
        on the same rows, only the row selection differs -- FOR ROWS WHOSE
        READS STAY INSIDE THE INPUT'S RIGHT EDGE.  At the bare right edge
        the two paths clamp differently (this path clamps a shifted read
        to the last lattice row; the dense path clamps to frame T-1, a
        frame that may be off the row grid), so callers must feed a window
        carrying >= right_context frames beyond the last row they keep --
        the recognizer's ``_run_step`` window always does (kaldi's own
        chunked-decode contract).  The left edge is not affected."""
        plan = self.decimation_plan(factor)
        T = features.shape[1]
        batch, t_len = features.shape[0], features.shape[1]
        iv = ivector[:, None, :].expand(-1, t_len, -1)

        # input row j of the lattice is anchored at input frame j*factor
        window_offset = 0
        num_rows = (T + factor - 1) // factor  # ceil(T / factor)

        nodes_out: Dict[str, torch.Tensor] = {}  # full-rate values
        lat_out: Dict[str, torch.Tensor] = {}  # lattice values

        def lattice(name: str) -> torch.Tensor:
            if name not in lat_out:
                lat_out[name] = nodes_out[name][:, window_offset::factor, :]
            return lat_out[name]

        for node in self.nodes:
            if node.kind == "input":
                if node.name == "input":
                    nodes_out[node.name] = features
                elif node.name == "ivector":
                    nodes_out[node.name] = iv
                else:
                    raise ValueError(f"unknown input node {node.name!r}")
            elif node.kind == "dim-range":
                if node.name in plan.top:
                    lat_out[node.name] = lattice(node.source)[
                        :, :, node.dim_offset : node.dim_offset + node.dim
                    ]
                else:
                    src = nodes_out[node.source]
                    nodes_out[node.name] = src[
                        :, :, node.dim_offset : node.dim_offset + node.dim
                    ]
            elif node.kind == "component":
                mod = self.component_modules[self._name_to_idx[node.component]]
                if node.name in plan.top:
                    x = self._eval_tree_lat(
                        parse_descriptor(node.descriptor), lattice, factor
                    )
                    lat_out[node.name] = (
                        mod(x, row_step=factor) if isinstance(mod, Tdnn) else mod(x)
                    )
                else:
                    x = self._eval_descriptor(
                        node.descriptor, nodes_out, batch, t_len, features, ivector
                    )
                    nodes_out[node.name] = mod(x)
            elif node.kind == "output":
                x = self._eval_tree_lat(
                    parse_descriptor(node.descriptor), lattice, factor
                )
                return x[:, :num_rows, :]
            else:
                raise ValueError(f"bad node kind {node.kind!r}")
        raise ValueError("no output node evaluated")

    # -- save / load

    def export_args(self) -> dict:
        """The 'args' dict stored in am.pt alongside the state_dict.

        Everything needed to reconstruct the module from scratch: the parsed
        network as plain (picklable) python objects.
        """
        return {
            "input_dim": self.spec.input_dim,
            "ivector_dim": self.spec.ivector_dim,
            "output_dim": self.spec.output_dim,
            "left_context": self.spec.left_context,
            "right_context": self.spec.right_context,
            "nodes": [vars(n) for n in self.spec.nodes],
            "components": [
                {"name": c.name, "type": c.type, "params": c.params}
                for c in self.spec.components
            ],
        }

    @staticmethod
    def from_export_args(args: dict) -> "Nnet3Torch":
        """Reconstruct the model from the dict saved by export_args()."""
        spec = Nnet3Spec(
            components=[Component(**c) for c in args["components"]],
            nodes=[Node(**n) for n in args["nodes"]],
            input_dim=args["input_dim"],
            ivector_dim=args["ivector_dim"],
            output_dim=args["output_dim"],
            left_context=args["left_context"],
            right_context=args["right_context"],
        )
        return Nnet3Torch(spec)


class VoskAmWrapper(nn.Module):
    """The exported unit: features + i-vector -> output log-probs (one row
    per `factor` input frames).

    The net is PURE: the caller supplies any context the receptive field
    needs (streams replicate edge frames -- kaldi's edge rule), and the
    output keeps the input's rows ``::factor`` (kaldi anchors decoder row
    k at input frame ``k*factor``).  The subsampling is computed *inside*
    the network evaluation on that row lattice (see DecimationPlan): the
    decimated zone computes only the rows the decoder reads -- like
    kaldi's sparse computation request -- at ~1/factor the rows, with
    values identical to slicing the full-rate output.  Dynamic T
    (variable-length input) is preserved.
    """

    def __init__(self, net: Nnet3Torch, frame_subsampling_factor: int = 3):
        super().__init__()
        self.net = net
        self.frame_subsampling_factor = frame_subsampling_factor
        self.plan = (
            net.decimation_plan(frame_subsampling_factor)
            if frame_subsampling_factor > 1
            else None
        )

    def forward(self, features: torch.Tensor, ivector: torch.Tensor) -> torch.Tensor:
        if self.plan is not None:
            return self.net.forward_lattice(
                features, ivector, self.frame_subsampling_factor
            )
        full = self.net(features, ivector)  # [B, T, num_pdfs]
        return full[:, :: self.frame_subsampling_factor, :]
