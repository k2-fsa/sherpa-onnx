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

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# nnet3/ is imported as a plain subpackage (scripts are run with scripts/vosk
# as the working directory / on sys.path; see run.sh and the notes).
from nnet3.components import (
    IDENTITY_TYPES,
    Component,
    Node,
    Nnet3Spec,
    component_out_dim,
)
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


class RestrictedAttention(nn.Module):
    """kaldi RestrictedAttentionComponent (nnet-attention-component.cc).

    Windowed multi-head attention over time: per head the row holds
    (key, value, query) with query_dim = key_dim + context_dim and
    context_dim = num_left + 1 + num_right.  For output row t, context
    position o reads input row t + (o - num_left) * time_stride
    (zero-padded outside the utterance -- kaldi's rule for this
    component):

        logit(t, o) = key_scale * q_key(t) . k(t+o) + q_ctx(t, o)
        c(t, :)     = softmax_o logit(t, :)
        out(t)      = sum_o c(t, o) * v(t+o)   (+ c(t, :) if output_context)

    Rows outside the utterance are CLAMPED to the edge (kaldi's rule,
    verified row-by-row against nnet3-compute).  The num_left/right_inputs_required
    fields only affect the compiled receptive field, not the math.
    """

    def __init__(self, c: Component):
        super().__init__()
        p = c.params
        self.num_heads = int(p["NumHeads"])
        self.key_dim = int(p["KeyDim"])
        self.value_dim = int(p["ValueDim"])
        self.num_left = int(p["NumLeftInputs"])
        self.num_right = int(p["NumRightInputs"])
        self.time_stride = int(p["TimeStride"])
        self.output_context = bool(p["OutputContext"])
        self.key_scale = float(p["KeyScale"])
        self.context_dim = self.num_left + 1 + self.num_right

    def forward(self, x: torch.Tensor, row_step: int = 1) -> torch.Tensor:
        # x: [B, T, num_heads * (key + value + key + context)]
        B, T, _ = x.shape
        kd, vd = self.key_dim, self.value_dim
        cd = self.context_dim
        qd = kd + cd
        head_in = kd + vd + qd
        outs = []
        for h in range(self.num_heads):
            inp = x[:, :, h * head_in : (h + 1) * head_in]
            k = inp[:, :, :kd]
            v = inp[:, :, kd : kd + vd]
            q_key = inp[:, :, kd + vd : kd + vd + kd]
            q_ctx = inp[:, :, kd + vd + kd :]  # [B, T, cd]
            ks, vs = [], []
            for o in range(cd):
                off = (o - self.num_left) * self.time_stride
                step = off // row_step
                assert step * row_step == off, (off, row_step)
                # clamped gather of row t + step (kaldi clamps the
                # window to the utterance edges -- verified row-by-row
                # against nnet3-compute; zero-padding does NOT match)
                idx = torch.arange(step, T + step, device=x.device)
                idx = idx.clamp(0, T - 1)
                ks.append(k.index_select(1, idx))
                vs.append(v.index_select(1, idx))
            k_all = torch.stack(ks, dim=2)  # [B, T, cd, kd]
            v_all = torch.stack(vs, dim=2)  # [B, T, cd, vd]
            logits = (
                torch.einsum("btk,btck->btc", q_key, k_all) * self.key_scale + q_ctx
            )  # [B, T, cd]
            prob = torch.softmax(logits, dim=-1)
            out_v = torch.einsum("btc,btck->btk", prob, v_all)  # [B, T, vd]
            if self.output_context:
                out = torch.cat([out_v, prob], dim=-1)
            else:
                out = out_v
            outs.append(out)
        return torch.cat(outs, dim=-1)


class Normalize(nn.Module):
    """kaldi NormalizeComponent (nnet-normalize-component.cc): per-block
    RMS normalization at inference:

        p        = ||x_block||^2 / (block_dim * target_rms^2)   (floor 2^-66)
        y_block  = x_block / sqrt(p)

    AddLogStddev appends log(rms(x_block)) to each block."""

    FLOOR = 1.3552527156068805425e-20  # kaldi's 2^-66

    def __init__(self, c: Component):
        super().__init__()
        p = c.params
        self.input_dim = int(p["InputDim"])
        self.block_dim = int(p.get("BlockDim", self.input_dim))
        self.target_rms = float(p["TargetRms"])
        self.add_log_stddev = bool(p["AddLogStddev"])
        assert self.input_dim % self.block_dim == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]; D = num_blocks * block_dim, blocks consecutive
        B, T, D = x.shape
        nb = D // self.block_dim
        blocks = x.view(B, T, nb, self.block_dim)
        p = (blocks * blocks).sum(dim=-1) / (
            self.block_dim * self.target_rms * self.target_rms
        )  # [B, T, nb]
        p = p.clamp(min=self.FLOOR)
        scale = torch.rsqrt(p).unsqueeze(-1)  # [B, T, nb, 1]
        out = blocks * scale
        if self.add_log_stddev:
            # log(rms(x_block)) = 0.5 log p + log(target_rms) (kaldi)
            log_col = (0.5 * torch.log(p) + math.log(self.target_rms)).unsqueeze(-1)
            out = torch.cat([out, log_col], dim=-1)
        return out.reshape(B, T, -1)


class LstmNonlinearity(nn.Module):
    """kaldi LstmNonlinearityComponent (nnet-combined-component.cc;
    math per cudamatrix CpuComputeLstmNonlinearity).

    Per row: input [i, f, c, o, c_prev] (5 blocks of cell_dim) ->
    output [c_t, m_t] (2 blocks):

        i_t = sigmoid(i + w_ic*c_prev);  f_t = sigmoid(f + w_fc*c_prev)
        c_t = f_t*c_prev + i_t*tanh(c);  o_t = sigmoid(o + w_oc*c_t)
        m_t = o_t*tanh(c_t)

    Params is [3, cell_dim] = (w_ic, w_fc, w_oc).  The recurrent wiring
    (feeding c_prev / m_t around) lives in the node graph, so this
    module is per-row.
    """

    def __init__(self, c: Component):
        super().__init__()
        p = c.params
        if p.get("UseDropout", False):
            raise ValueError(
                f"{c.name}: LstmNonlinearityComponent with UseDropout is "
                "not supported at inference"
            )
        params = np.ascontiguousarray(p["Params"])  # [3, cell_dim]
        self.cell_dim = params.shape[1]
        self.weight = nn.Parameter(torch.from_numpy(params), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 5*cell]
        cd = self.cell_dim
        i = x[..., 0:cd]
        f = x[..., cd : 2 * cd]
        g = x[..., 2 * cd : 3 * cd]
        o = x[..., 3 * cd : 4 * cd]
        c_prev = x[..., 4 * cd : 5 * cd]
        w_ic, w_fc, w_oc = self.weight[0], self.weight[1], self.weight[2]
        i_t = torch.sigmoid(i + w_ic * c_prev)
        f_t = torch.sigmoid(f + w_fc * c_prev)
        c_t = f_t * c_prev + i_t * torch.tanh(g)
        o_t = torch.sigmoid(o + w_oc * c_t)
        m_t = o_t * torch.tanh(c_t)
        return torch.cat([c_t, m_t], dim=-1)


class BackpropTruncation(nn.Module):
    """kaldi BackpropTruncationComponent at inference: y = Scale * x
    (the clipping/zeroing of the truncated-BPTT machinery only affects
    backprop)."""

    def __init__(self, c: Component):
        super().__init__()
        self.scale = float(c.params.get("Scale", 1.0))
        self.out_dim = int(c.params["Dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


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
      * for each offset: input row t+dt (CLAMPED at time edges, like
        Tdnn), input height h_in = h_out*subsample + dh
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

        # time-shifted copies; rows outside the utterance CLAMP to the
        # edge (kaldi's rule -- verified row-by-row against nnet3-compute;
        # zero-padding is wrong at the time edges)
        idx = torch.arange(T, device=x.device)
        parts = []
        for dt, dh in self.offsets:
            src = idx + dt
            xt = x4.index_select(1, src.clamp(0, T - 1))
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
    if c.type == "RestrictedAttentionComponent":
        return RestrictedAttention(c)
    if c.type == "NormalizeComponent":
        return Normalize(c)
    if c.type == "LstmNonlinearityComponent":
        return LstmNonlinearity(c)
    if c.type == "BackpropTruncationComponent":
        return BackpropTruncation(c)
    if c.type == "RectifiedLinearComponent":
        return nn.ReLU()
    if c.type == "LogSoftmaxComponent":
        return LogSoftmax()
    raise ValueError(f"no torch module for component type {c.type!r} ({c.name})")


# ---------------------------------------------------------------------------
# Decimation plan -- compute only the rows the decoder reads
# ---------------------------------------------------------------------------


def is_ivector_index(tree) -> bool:
    """True for ReplaceIndex(<expr of the ivector input only>, t, v).

    kaldi's looped/online decodables rewrite ReplaceIndex(ivector, t, 0) to
    Round(ivector, chunk_size) (nnet-compile-looped.cc
    ModifyNnetIvectorPeriod): the i-vector of row t is the one of t's
    chunk.  The ivector input here is per frame ([B, T, S]; the caller
    repeats each chunk's i-vector over the chunk's frames), so the
    rewritten descriptor is simply "read row t" of the ivector input."""
    return (
        tree[0] == "replace_index"
        and tree[2] == "t"
        and descriptor_names(tree[1]) == {"ivector"}
    )


def check_ivector_rows(ivector: torch.Tensor, t_len: int) -> torch.Tensor:
    """The ivector input must be per frame: [B, T, S], T == features T."""
    if ivector.dim() != 3 or ivector.shape[1] != t_len:
        raise ValueError(
            f"ivector must be [B, T={t_len}, S] (one i-vector per input "
            f"frame), got {tuple(ivector.shape)}"
        )
    return ivector


def nnet_modulus(nodes: List[Node]) -> int:
    """kaldi's Nnet::Modulus(): the lcm over all descriptors of their
    time modulus (Round(x, t) -> t, Switch(a, b, ...) -> its number of
    sources; everything else 1).  kaldi's GetChunkSize rounds the
    looped decodable's frames_per_chunk up to a multiple of it."""

    def tree_mod(tree) -> int:
        kind = tree[0]
        if kind == "name" or kind == "const":
            return 1
        if kind in ("append", "sum"):
            return _lcm_all(tree_mod(x) for x in tree[1])
        if kind == "switch":
            return math.lcm(len(tree[1]), _lcm_all(tree_mod(x) for x in tree[1]))
        if kind == "scale":
            return tree_mod(tree[2])
        if kind == "round":
            return int(tree[2])  # kaldi: t_modulus_ (not combined)
        if kind == "failover":
            return math.lcm(tree_mod(tree[1]), tree_mod(tree[2]))
        if kind in ("offset", "replace_index", "if_defined"):
            return tree_mod(tree[1])
        raise ValueError(f"bad descriptor tree {tree!r}")

    ans = 1
    for n in nodes:
        if n.kind in ("component", "output"):
            ans = math.lcm(ans, tree_mod(parse_descriptor(n.descriptor)))
    return ans


def _lcm_all(values) -> int:
    ans = 1
    for v in values:
        ans = math.lcm(ans, v)
    return ans


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
    if c.type == "RestrictedAttentionComponent":
        s = int(c.params["TimeStride"])
        return {
            o * s
            for o in range(
                -int(c.params["NumLeftInputs"]), int(c.params["NumRightInputs"]) + 1
            )
        }
    return {0}  # per-row components (affine, batchnorm, relu, normalize, ...)


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
    if c.type == "RestrictedAttentionComponent":
        cd = int(p["NumLeftInputs"]) + 1 + int(p["NumRightInputs"])
        return float(int(p["NumHeads"]) * cd * (int(p["KeyDim"]) + int(p["ValueDim"])))
    return 0.0


def recurrent_names(nodes: List[Node]) -> set:
    """Names of nodes that take part in a time recurrence (a cycle in the
    node graph: kaldi wires e.g. c_prev = Offset(c, -1) around
    LstmNonlinearityComponent)."""
    by_name = {n.name: n for n in nodes}
    refs = {}
    for n in nodes:
        if n.kind in ("component", "output"):
            refs[n.name] = descriptor_names(parse_descriptor(n.descriptor))
        elif n.kind == "dim-range":
            refs[n.name] = [n.source]
        else:
            refs[n.name] = []
    cyc = set()

    def reaches_self(start: str) -> bool:
        seen, stack = set(), list(refs.get(start, ()))
        while stack:
            x = stack.pop()
            if x == start:
                return True
            if x in seen:
                continue
            seen.add(x)
            stack.extend(refs.get(x, ()))
        return False

    for n in nodes:
        if reaches_self(n.name):
            cyc.add(n.name)
    return cyc


def first_computable_rows(nodes: List[Node], components: List[Component]) -> dict:
    """kaldi's IsComputable, per node: the first row computable when the
    input rows start at row 0 (kaldi only feeds input rows it has --
    nothing before the first fed row exists, and intermediate nodes do
    NOT clamp).  IfDefined adds no constraint (it yields zeros when its
    argument is not computable), which is what breaks recurrence cycles:
    a recurrent loop's blank (zero-state) rows are exactly the rows
    before the first computable row of the loop's non-recurrent input.
    Each recurrent cluster therefore has its OWN origin -- deeper
    clusters start later."""
    comp_by_name = {c.name: c for c in components}
    neg = -(10**9)
    lo = {n.name: neg for n in nodes}
    for n in nodes:
        if n.kind == "input":
            lo[n.name] = 0 if n.name == "input" else neg  # ivector: any row

    def tree_lo(tree):
        kind = tree[0]
        if kind == "name":
            return lo[tree[1]]
        if kind in ("append", "sum"):
            return max(tree_lo(x) for x in tree[1])
        if kind == "switch":
            return max(tree_lo(x) for x in tree[1])
        if kind == "scale":
            return tree_lo(tree[2])
        if kind == "offset":
            v = tree_lo(tree[1])
            return v if v == neg else v - tree[2]
        if kind == "round":
            v = tree_lo(tree[1])
            return v if v == neg else -((-v) // tree[2]) * tree[2]
        if kind == "failover":
            return min(tree_lo(tree[1]), tree_lo(tree[2]))
        if kind in ("if_defined", "const", "replace_index"):
            return neg
        raise ValueError(f"bad descriptor tree {tree!r}")

    # monotone fixed point (acyclic once IfDefined edges are dropped)
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n.kind == "component":
                v = tree_lo(parse_descriptor(n.descriptor))
                deps = _component_time_deps(comp_by_name[n.component])
                if v != neg:
                    v -= min(deps)
            elif n.kind == "dim-range":
                v = lo[n.source]
            elif n.kind == "output":
                v = tree_lo(parse_descriptor(n.descriptor))
            else:
                continue
            if v > lo[n.name]:
                lo[n.name] = v
                changed = True
    return lo


def recurrent_cycles(nodes: List[Node]) -> dict:
    """recurrent node name -> the names on its cycle (its strongly-
    connected component in the node graph)."""
    refs = {}
    for n in nodes:
        if n.kind in ("component", "output"):
            refs[n.name] = descriptor_names(parse_descriptor(n.descriptor))
        elif n.kind == "dim-range":
            refs[n.name] = [n.source]
        else:
            refs[n.name] = []

    def reach(start):
        seen, stack = set(), list(refs.get(start, ()))
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.add(x)
                stack.extend(refs.get(x, ()))
        return seen

    reached = {n.name: reach(n.name) for n in nodes}
    return {
        a: {b for b in reached[a] if a in reached[b]}
        for a in reached
        if a in reached[a]
    }


def descriptor_row_refs(tree, base=0, out=None):
    """[(name, total_offset)] the descriptor tree reads (IfDefined etc.
    included)."""
    if out is None:
        out = []
    kind = tree[0]
    if kind == "name":
        out.append((tree[1], base))
    elif kind == "offset":
        descriptor_row_refs(tree[1], base + tree[2], out)
    elif kind in ("append", "sum"):
        for x in tree[1]:
            descriptor_row_refs(x, base, out)
    elif kind == "scale":
        descriptor_row_refs(tree[2], base, out)
    elif kind == "replace_index":
        descriptor_row_refs(tree[1], tree[3], out)
    elif kind in ("if_defined", "round", "failover", "switch"):
        kids = (
            tree[1]
            if kind == "switch"
            else ([tree[1], tree[2]] if kind == "failover" else [tree[1]])
        )
        for x in kids:
            descriptor_row_refs(x, base, out)
    elif kind == "const":
        pass
    else:
        raise ValueError(f"bad descriptor tree {tree!r}")
    return out


def recurrent_state_layout(nodes: List[Node]):
    """The streaming state of a recurrent net: (rows, names).

    names: the recurrent nodes read back across time inside their own
    cycle (e.g. lstm1.c_trunc via IfDefined(Offset(lstm1.c_trunc, -3))),
    sorted; rows: the largest such time lag (3 for a clockwork LSTM).
    The state holds, for each name, the `rows` rows just before the
    node's first computable row -- exactly what the loop reads there.
    (rows, []) = (0, []) for a net without recurrence."""
    cycles = recurrent_cycles(nodes)
    lag = {}
    for n in nodes:
        if n.name not in cycles or n.kind not in ("component", "output"):
            continue
        for name, off in descriptor_row_refs(parse_descriptor(n.descriptor)):
            if name in cycles[n.name] and off < 0:
                lag[name] = max(lag.get(name, 0), -off)
    return (max(lag.values()) if lag else 0), sorted(lag)


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
        self._nodes_by_name = {n.name: n for n in self.nodes}
        # nodes taking part in a time recurrence (evaluated row by row)
        self._recurrent = recurrent_names(self.nodes)
        # first computable row per node (recurrent loops start blank below it)
        self._first_row = first_computable_rows(self.nodes, spec.components)
        # recurrent node -> all members of its cycle (strongly-connected
        # component), flushed together by _eval_recurrent
        self._cycle_of = recurrent_cycles(self.nodes)
        # streaming state (recurrent nets): `state_rows` rows of each
        # fed-back node, concatenated on the feature axis -> [B, rows, dim]
        self.state_rows, names = recurrent_state_layout(self.nodes)
        self.state_layout = []  # (name, dim_offset, dim)
        off = 0
        for name in names:
            if self._first_row[name] < self.state_rows:
                raise NotImplementedError(
                    f"{name}: first computable row {self._first_row[name]} < "
                    f"state rows {self.state_rows}"
                )
            d = self._node_out_dim(name)
            self.state_layout.append((name, off, d))
            off += d
        self.state_dim = off
        self._state_slot = {n: (o, d) for n, o, d in self.state_layout}
        self._state_in = None
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
            if is_ivector_index(tree):
                return x  # per-frame i-vector rows (see is_ivector_index)
            _, _expr, index_name, value = tree
            if index_name != "t":
                raise ValueError(f"unsupported ReplaceIndex axis {index_name!r}")
            t = max(0, min(value, x.shape[1] - 1))
            return x[:, t : t + 1, :].expand(-1, t_len, -1)
        if kind in ("const", "round", "switch", "if_defined", "failover"):
            # row-dependent forms: evaluate row by row (rare)
            cache: Dict[tuple, torch.Tensor] = {}
            busy = set()
            return torch.cat(
                [
                    self._eval_tree_row(
                        tree,
                        t,
                        cache,
                        busy,
                        nodes_out,
                        batch,
                        t_len,
                        features,
                        ivector,
                    )
                    for t in range(t_len)
                ],
                dim=1,
            )
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

    # -- recurrent evaluation (row by row) -------------------------------

    def _node_out_dim(self, name: str) -> int:
        node = self._nodes_by_name[name]
        if node.kind == "dim-range":
            return node.dim
        return component_out_dim(self.spec.component(node.component))

    def _tree_defined(self, tree, t: int, t_len: int) -> bool:
        """kaldi 'IsComputable' for one row: every row the tree reads is
        inside the utterance (rows outside are UNDEFINED -- IfDefined's
        zero fallback applies)."""
        kind = tree[0]
        if kind == "name":
            return 0 <= t < t_len
        if kind == "offset":
            return self._tree_defined(tree[1], t + tree[2], t_len)
        if kind == "replace_index":
            return self._tree_defined(tree[1], tree[3], t_len)
        if kind in ("append", "sum"):
            return all(self._tree_defined(x, t, t_len) for x in tree[1])
        if kind == "scale":
            return self._tree_defined(tree[2], t, t_len)
        if kind == "const":
            return True
        if kind == "round":
            return self._tree_defined(tree[1], (t // tree[2]) * tree[2], t_len)
        if kind == "switch":
            return self._tree_defined(tree[1][t % len(tree[1])], t, t_len)
        if kind == "if_defined":
            return True  # always yields a value (zeros as the fallback)
        if kind == "failover":
            return self._tree_defined(tree[1], t, t_len) or self._tree_defined(
                tree[2], t, t_len
            )
        raise ValueError(f"bad descriptor tree {tree!r}")

    def _tree_dim(self, tree) -> int:
        kind = tree[0]
        if kind == "name":
            return self._node_out_dim(tree[1])
        if kind == "append":
            return sum(self._tree_dim(x) for x in tree[1])
        if kind in ("sum", "if_defined", "failover", "round", "offset"):
            return self._tree_dim(tree[1])
        if kind == "switch":
            return self._tree_dim(tree[1][0])
        if kind == "scale":
            return self._tree_dim(tree[2])
        if kind == "const":
            return tree[2]
        if kind == "replace_index":
            return self._tree_dim(tree[1])
        raise ValueError(f"bad descriptor tree {tree!r}")

    def _eval_row(
        self, name, t, cache, busy, nodes_out, batch, t_len, features, ivector
    ):
        """One row [B, 1, D] of a node -- used for recurrent groups (see
        _eval_recurrent).  Rows of already-evaluated nodes are gathered
        with the usual edge clamp.  Rows of a recurrent node below its
        first computable row (first_computable_rows) are BLANK (zeros):
        kaldi cannot compute them, so IfDefined reads zeros there -- the
        zero-initialized LSTM state.  They are still evaluated (then
        zeroed) so every member of the cycle gets filled, exactly like
        the flat module's loop."""
        if name in nodes_out:
            seq = nodes_out[name]
            tt = t if 0 <= t < seq.shape[1] else (0 if t < 0 else seq.shape[1] - 1)
            return seq[:, tt : tt + 1, :]
        if t < 0:
            return features.new_zeros(batch, 1, self._node_out_dim(name))
        key = (name, t)
        if key in cache:
            return cache[key]
        if key in busy:
            raise ValueError(
                f"recurrence with non-negative offset at {name!r} row {t} "
                "(only causal loops are supported)"
            )
        busy.add(key)
        node = self._nodes_by_name[name]
        if node.kind == "dim-range":
            row = self._eval_row(
                node.source, t, cache, busy, nodes_out, batch, t_len, features, ivector
            )[..., node.dim_offset : node.dim_offset + node.dim]
        elif node.kind == "component":
            x = self._eval_tree_row(
                parse_descriptor(node.descriptor),
                t,
                cache,
                busy,
                nodes_out,
                batch,
                t_len,
                features,
                ivector,
            )
            mod = self.component_modules[self._name_to_idx[node.component]]
            row = mod(x)
        else:
            raise ValueError(f"recurrent node {name!r}: unsupported kind {node.kind!r}")
        first = self._first_row[name]
        if t < first:
            slot = self._state_slot.get(name)
            if (
                self._state_in is not None
                and slot is not None
                and t >= first - self.state_rows
            ):
                # the carried streaming state (rows just before the loop
                # origin, computed by the previous window)
                k = t - (first - self.state_rows)
                row = self._state_in[:, k : k + 1, slot[0] : slot[0] + slot[1]]
            else:
                # not computable in kaldi: a blank row (zero LSTM state)
                row = torch.zeros_like(row)
        busy.discard(key)
        cache[key] = row
        return row

    def _eval_tree_row(
        self, tree, t, cache, busy, nodes_out, batch, t_len, features, ivector
    ):
        kind = tree[0]
        if kind == "name":
            return self._eval_row(
                tree[1], t, cache, busy, nodes_out, batch, t_len, features, ivector
            )
        if kind == "append":
            return torch.cat(
                [
                    self._eval_tree_row(
                        x, t, cache, busy, nodes_out, batch, t_len, features, ivector
                    )
                    for x in tree[1]
                ],
                dim=-1,
            )
        if kind == "sum":
            out = self._eval_tree_row(
                tree[1][0], t, cache, busy, nodes_out, batch, t_len, features, ivector
            )
            for x in tree[1][1:]:
                out = out + self._eval_tree_row(
                    x, t, cache, busy, nodes_out, batch, t_len, features, ivector
                )
            return out
        if kind == "scale":
            return tree[1] * self._eval_tree_row(
                tree[2], t, cache, busy, nodes_out, batch, t_len, features, ivector
            )
        if kind == "offset":
            return self._eval_tree_row(
                tree[1],
                t + tree[2],
                cache,
                busy,
                nodes_out,
                batch,
                t_len,
                features,
                ivector,
            )
        if kind == "replace_index":
            _, expr, index_name, value = tree
            if index_name != "t":
                raise ValueError(f"unsupported ReplaceIndex axis {index_name!r}")
            return self._eval_tree_row(
                expr,
                t if is_ivector_index(tree) else value,
                cache,
                busy,
                nodes_out,
                batch,
                t_len,
                features,
                ivector,
            )
        if kind == "const":
            return features.new_full((batch, 1, tree[2]), tree[1])
        if kind == "round":
            return self._eval_tree_row(
                tree[1],
                (t // tree[2]) * tree[2],
                cache,
                busy,
                nodes_out,
                batch,
                t_len,
                features,
                ivector,
            )
        if kind == "switch":
            return self._eval_tree_row(
                tree[1][t % len(tree[1])],
                t,
                cache,
                busy,
                nodes_out,
                batch,
                t_len,
                features,
                ivector,
            )
        if kind == "if_defined":
            if self._tree_defined(tree[1], t, t_len):
                return self._eval_tree_row(
                    tree[1],
                    t,
                    cache,
                    busy,
                    nodes_out,
                    batch,
                    t_len,
                    features,
                    ivector,
                )
            return features.new_zeros(batch, 1, self._tree_dim(tree[1]))
        if kind == "failover":
            use = tree[1] if self._tree_defined(tree[1], t, t_len) else tree[2]
            return self._eval_tree_row(
                use, t, cache, busy, nodes_out, batch, t_len, features, ivector
            )
        raise ValueError(f"bad descriptor tree {tree!r}")

    def _eval_recurrent(self, node, nodes_out, batch, t_len, features, ivector):
        """Evaluate a node that is part of a time recurrence (kaldi wires
        e.g. c_prev = Offset(c, -1) around LstmNonlinearityComponent):
        rows are computed sequentially, on demand across the whole
        cycle."""
        cache: Dict[tuple, torch.Tensor] = {}
        busy = set()
        rows = [
            self._eval_row(
                node.name, t, cache, busy, nodes_out, batch, t_len, features, ivector
            )
            for t in range(t_len)
        ]
        nodes_out[node.name] = torch.cat(rows, dim=1)
        # flush every node the recursion filled along the way; rows that
        # were never demanded (e.g. the last row of a dim-range sliced out
        # of the group) are computed here
        for name in sorted(self._cycle_of[node.name]):
            if name not in nodes_out:
                nodes_out[name] = torch.cat(
                    [
                        self._eval_row(
                            name,
                            t,
                            cache,
                            busy,
                            nodes_out,
                            batch,
                            t_len,
                            features,
                            ivector,
                        )
                        for t in range(t_len)
                    ],
                    dim=1,
                )

    # -- forward ---------------------------------------------------------

    def forward(self, features: torch.Tensor, ivector: torch.Tensor) -> torch.Tensor:
        """
        :param features: [B, T, input_dim] float32 (e.g. 40 MFCC)
        :param ivector:  [B, T, ivector_dim] float32, one i-vector per
                         input frame (kaldi's online decoders: the
                         i-vector of each frame's chunk, see
                         is_ivector_index)
        :return: [B, T, num_pdfs] log-probs, one row per input frame

        The net is PURE: it computes exactly the rows it is given.  Edge
        semantics are the caller's job -- to match kaldi at the edges of
        an utterance, feed replicated edge frames beyond the real input
        (kaldi clamps out-of-range input indexes to the edge; the
        recognizer's streams do this in get_frames), i.e. left_context
        rows before the real utterance and right_context rows after it.

        Recurrent loops (kaldi wires e.g. c_prev = Offset(c, -3) around
        LstmNonlinearityComponent) start with BLANK rows (zeros -- the
        zero-initialized cell state) below the first COMPUTABLE row of
        each loop node, counted from the first fed row (see
        first_computable_rows).  With the caller padding by left_context
        this equals kaldi's unchunked computation (nnet3-compute
        --frames-per-chunk >= T, and the looped online decodable).
        """
        return self.node_outputs(features, ivector)["output"]

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        """The streaming state at the start of an utterance: zeros
        [batch, state_rows, state_dim] (== the blank rows of kaldi)."""
        return torch.zeros(batch, self.state_rows, self.state_dim)

    def forward_streaming(
        self,
        features: torch.Tensor,
        ivector: torch.Tensor,
        state: torch.Tensor,
        shift: int,
    ):
        """One window of a stream with the recurrent state carried over.

        :param features: [B, W, input_dim] -- the window, row 0 = the
            window's first frame (left context included)
        :param state: [B, state_rows, state_dim] from the previous window
            (initial_state() for the first one)
        :param shift: input frames from this window's start to the next
            window's start
        :return: (log-probs [B, W, num_pdfs], state for the next window)

        Each recurrent loop starts at its first computable row of the
        window (fixed: it only depends on the window start); the rows
        just before it come from `state` instead of being blank.  The
        returned state is those rows of the window that starts `shift`
        frames later -- for windows laid out like a one-shot run's
        (left context + new frames + right context) the rows are fully
        computable here, so chaining windows reproduces the one-shot
        computation (kaldi's looped decodable carries the same state).
        """
        nodes = self.node_outputs(features, ivector, state)
        return nodes["output"], self._state_rows_at(nodes, shift)

    def _state_rows_at(self, nodes: dict, shift: int) -> torch.Tensor:
        parts = []
        for name, _off, _dim in self.state_layout:
            seq = nodes[name]
            start = self._first_row[name] + shift - self.state_rows
            idx = (torch.arange(self.state_rows) + start).clamp(0, seq.shape[1] - 1)
            parts.append(seq.index_select(1, idx))
        return torch.cat(parts, dim=-1)

    def node_outputs(
        self,
        features: torch.Tensor,
        ivector: torch.Tensor,
        state: torch.Tensor = None,
    ) -> dict:
        """Every node's tensor (debugging aid; forward() returns the
        'output' entry).  `state`: see forward_streaming (None = blank)."""
        self._state_in = state
        batch, t_len = features.shape[0], features.shape[1]
        iv = check_ivector_rows(ivector, t_len)  # [B, T, S]

        nodes_out: Dict[str, torch.Tensor] = {}
        for node in self.nodes:
            if node.name in nodes_out:
                continue  # filled by a recurrent group
            if node.kind == "input":
                if node.name == "input":
                    nodes_out[node.name] = features
                elif node.name == "ivector":
                    nodes_out[node.name] = iv
                else:
                    raise ValueError(f"unknown input node {node.name!r}")
            elif node.kind == "component":
                if any(
                    n not in nodes_out
                    for n in descriptor_names(parse_descriptor(node.descriptor))
                ):
                    self._eval_recurrent(
                        node, nodes_out, batch, t_len, features, ivector
                    )
                    continue
                x = self._eval_descriptor(
                    node.descriptor, nodes_out, batch, t_len, features, ivector
                )
                nodes_out[node.name] = self.component_modules[
                    self._name_to_idx[node.component]
                ](x)
            elif node.kind == "dim-range":
                if node.source not in nodes_out:
                    self._eval_recurrent(
                        node, nodes_out, batch, t_len, features, ivector
                    )
                    continue
                src = nodes_out[node.source]
                nodes_out[node.name] = src[
                    :, :, node.dim_offset : node.dim_offset + node.dim
                ]
            elif node.kind == "output":
                nodes_out[node.name] = self._eval_descriptor(
                    node.descriptor, nodes_out, batch, t_len, features, ivector
                )
        return nodes_out

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
        iv = check_ivector_rows(ivector, t_len)

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
                        mod(x, row_step=factor)
                        if isinstance(mod, (Tdnn, RestrictedAttention))
                        else mod(x)
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
            if frame_subsampling_factor > 1 and not net._recurrent
            else None  # recurrent nets run full-rate (see recurrent_names)
        )

    def forward(self, features: torch.Tensor, ivector: torch.Tensor) -> torch.Tensor:
        if self.plan is not None:
            return self.net.forward_lattice(
                features, ivector, self.frame_subsampling_factor
            )
        full = self.net(features, ivector)  # [B, T, num_pdfs]
        return full[:, :: self.frame_subsampling_factor, :]
