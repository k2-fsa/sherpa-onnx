# sherpa-onnx/scripts/vosk/nnet3/components.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Dataclasses describing a parsed kaldi nnet3 network (AmNnetSimple), plus the
# per-component parameter schemas used by mdl_reader.py.  Inference-time
# semantics only: training-only fields are recognized, consumed and dropped.
#
# Component semantics at inference (kaldi references):
#
#   TdnnComponent            nnet3/nnet-tdnn-component.cc + nnet-convolutional-component.h
#                            y(t) = bias + sum_i W_i x(t + offset_i)
#                            LinearParams layout: [out_dim, in_dim * num_offsets],
#                            column-block i belongs to TimeOffsets[i]
#   BatchNormComponent       nnet3/nnet-normalize-component.cc (ComputeDerived,
#                            test mode): y = x * scale + offset with
#                              mean  = StatsMean
#                              var   = StatsVar          (floored at 0)
#                              scale = (var + eps)^-0.5 * TargetRms
#                              offset= -mean * scale
#   NaturalGradientAffine    nnet-simple-component.cc: plain affine at
#                            inference (the natural-gradient machinery is
#                            training-only)
#   LinearComponent          nnet-simple-component.cc: affine without bias
#   FixedAffineComponent     nnet-simple-component.cc: y = W x + b with a
#                            constant W (e.g. the idct layer)
#   LogSoftmaxComponent      log-softmax over the feature dimension
#   RectifiedLinearComponent ReLU
#   NoOpComponent            identity (BackpropScale is training-only)
#   GeneralDropoutComponent  identity at inference (dropout proportions etc.
#                            only affect training)
#   SpecAugmentTimeMaskComponent  identity at inference (data augmentation)
#   TimeHeightConvolutionComponent  2-D convolution over (time x height),
#                            zero-padded at the edges (large-en; see the
#                            torch_model.TimeHeightConvolution docstring)
#   PermuteComponent         nnet-simple-component.cc: fixed column
#                            permutation (large-en)
#   RestrictedAttentionComponent  nnet3/nnet-attention-component.cc:
#                            windowed multi-head attention over time
#                            (large-en, fr-linto; math in the schema
#                            comment below)
#   NormalizeComponent       nnet3/nnet-normalize-component.cc: per-block
#                            RMS normalization (aspire)
#   LstmNonlinearityComponent  nnet3/nnet-combined-component.cc: LSTM cell
#                            nonlinearity (gigaspeech; math in the schema
#                            comment below)
#   BackpropTruncationComponent  nnet3/nnet-general-component.cc: y =
#                            Scale * x at inference (gigaspeech wires it
#                            on the recurrent edge)
#
# See notes.md in this directory for the design write-up.

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Parsed network description
# ---------------------------------------------------------------------------


@dataclass
class Component:
    """One component of the network (name + type + parsed parameters)."""

    name: str  # e.g. 'tdnn1.affine'
    type: str  # e.g. 'NaturalGradientAffineComponent'
    params: dict = field(
        default_factory=dict
    )  # tag -> np.ndarray | float | bool | list


@dataclass
class Node:
    """One entry of the network's node graph (the config section of the dump).

    kind:
      'input'      input-node (features / ivector)
      'component'  component-node: evaluate `component` on `descriptor`
      'dim-range'  dim-range-node: slice dims [dim_offset, dim_offset+dim)
                   of node `source`
      'output'     output-node: the network output; `descriptor` names the
                   feeding node
    """

    name: str
    kind: str
    component: str = ""  # component name (kind == 'component')
    descriptor: str = ""  # input expression (kind == 'component' | 'output')
    source: str = ""  # source node (kind == 'dim-range')
    dim_offset: int = 0
    dim: int = 0


@dataclass
class Nnet3Spec:
    """The whole parsed network, ready to build a PyTorch model from."""

    components: List[Component]
    nodes: List[Node]
    input_dim: int  # 'input' node dim (e.g. 40 MFCC)
    ivector_dim: int  # 'ivector' node dim (0 if the model has none)
    output_dim: int  # 'output' dim == num_pdfs
    left_context: int  # receptive field from the dump tail
    right_context: int

    def component(self, name: str) -> Component:
        """Look up a component by name (raises KeyError)."""
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"component not found: {name}")


# ---------------------------------------------------------------------------
# Parameter schemas
# ---------------------------------------------------------------------------
#
# tag kinds:  'mat'       -> np.float32 matrix, rows = text lines
#             'vec'       -> np.float32 vector (single line)
#             'offsets'   -> list[int] (a single matrix row, TimeOffsets)
#             'num'       -> float
#             'bool'      -> 'T'/'F' -> bool
#             'flag'      -> presence flag: kaldi writes the tag with no
#                           value when true (e.g. <Continuous>); absence = False
#             'struct'    -> structural opener with no value (e.g. <Model>
#                           wrapping nested parameter tags)
#             'pairs'     -> one matrix row of ints, grouped in pairs
#                           (THConv's <Offsets> = [(time, height), ...])
#             'ignore'    -> consumed and dropped (training-only state)
#
# TRAINING_ONLY_TAGS are accepted for every component type: they are the
# optimizer/normalization bookkeeping kaldi writes after the parameters
# (natural-gradient state, running objective averages, self-repair counters,
# ...).  Anything *not* in the schemas or here raises — a parameter we do not
# understand must fail the export, never be silently skipped.

TRAINING_ONLY_TAGS = {
    # optimizer / natural-gradient state
    "MaxChange": "ignore",
    "L2Regularize": "ignore",
    "LearningRate": "ignore",
    "LearningRateFactor": "ignore",
    "NumSamplesHistory": "ignore",
    "UpdatePeriod": "ignore",
    "Alpha": "ignore",
    "AlphaInOut": "ignore",
    "RankIn": "ignore",
    "RankOut": "ignore",
    "RankInOut": "ignore",
    "OrthonormalConstraint": "ignore",
    "UseNaturalGradient": "ignore",
    # running statistics / diagnostics
    "ValueAvg": "ignore",
    "DerivAvg": "ignore",
    "Count": "ignore",
    "OderivRms": "ignore",
    "OderivCount": "ignore",
    "NumDimsSelfRepaired": "ignore",
    "NumDimsProcessed": "ignore",
    "SelfRepairScale": "ignore",
}

# Semantic (inference-relevant) parameters per component type.  The shared
# training-only tags are merged in automatically below.
_SEMANTIC_SCHEMAS = {
    # y = W x + b, constant W (e.g. the idct layer at the net's input)
    "FixedAffineComponent": {"LinearParams": "mat", "BiasParams": "vec"},
    # frozen affine in test mode (formula in the header comment above)
    "BatchNormComponent": {
        "Dim": "num",
        "BlockDim": "num",
        "Epsilon": "num",
        "TargetRms": "num",
        "TestMode": "bool",
        "StatsMean": "vec",
        "StatsVar": "vec",
    },
    # y(t) = b + sum_i W_i x(t+offset_i); W_i = LinearParams[:, i*Din:(i+1)*Din]
    "TdnnComponent": {
        "TimeOffsets": "offsets",
        "LinearParams": "mat",
        "BiasParams": "vec",
    },
    # plain affine at inference
    "NaturalGradientAffineComponent": {"LinearParams": "mat", "BiasParams": "vec"},
    # affine without bias
    "LinearComponent": {"Params": "mat"},
    # log-softmax over the feature dim (the network output)
    "LogSoftmaxComponent": {"Dim": "num"},
    "RectifiedLinearComponent": {
        "Dim": "num",
        "BlockDim": "num",
        "SelfRepairLowerThreshold": "ignore",
    },
    "NoOpComponent": {"Dim": "num", "BackpropScale": "num"},
    "GeneralDropoutComponent": {
        "Dim": "num",
        "BlockDim": "num",
        "TimePeriod": "num",
        "DropoutProportion": "num",
        "SpecAugmentMaxProportion": "num",
        "Continuous": "flag",
    },
    "SpecAugmentTimeMaskComponent": {
        "Dim": "num",
        "ZeroedProportion": "num",
        "TimeMaskMaxFrames": "num",
    },
    "TimeHeightConvolutionComponent": {
        "MaxChange": "ignore",
        "L2Regularize": "ignore",
        "LearningRate": "ignore",
        "Model": "struct",
        "ConvolutionModel": "struct",
        "NumFiltersIn": "num",
        "NumFiltersOut": "num",
        "HeightIn": "num",
        "HeightOut": "num",
        "HeightSubsampleOut": "num",
        "Offsets": "pairs",
        "RequiredTimeOffsets": "ignore",
        "LinearParams": "mat",
        "BiasParams": "vec",
        "MaxMemoryMb": "ignore",
        "UseNaturalGradient": "ignore",
        "NumMinibatchesHistory": "ignore",
        "AlphaInOut": "ignore",
        "RankInOut": "ignore",
    },
    "PermuteComponent": {"ColumnMap": "offsets"},
    # windowed multi-head attention over time
    # (nnet3/nnet-attention-component.cc + attention.cc):
    #   context_dim = NumLeftInputs + 1 + NumRightInputs; per head the row
    #   holds (key, value, query) with query_dim = key_dim + context_dim;
    #   for output row t, context position o reads input row
    #   t + (o - NumLeftInputs) * TimeStride (zero-padded at the edges):
    #     logit(t, o) = KeyScale * q_key(t).k(t+o) + q_ctx(t, o)
    #     c(t, :)     = softmax_o logit(t, :)
    #     out(t)      = sum_o c(t, o) * v(t+o)  (+ c(t, :) if OutputContext)
    #   rows outside the utterance CLAMP to the edge (verified row-by-row
    #   against nnet3-compute -- NOT zero-padded).  NumLeft/RightInputsRequired
    #   only affect the compiled receptive field, not the math.
    #   <StatsCount>/<EntropyStats>/<PosteriorStats> are training statistics.
    "RestrictedAttentionComponent": {
        "NumHeads": "num",
        "KeyDim": "num",
        "ValueDim": "num",
        "NumLeftInputs": "num",
        "NumRightInputs": "num",
        "TimeStride": "num",
        "NumLeftInputsRequired": "num",
        "NumRightInputsRequired": "num",
        "OutputContext": "bool",
        "KeyScale": "num",
        "StatsCount": "ignore",
        "EntropyStats": "vec",
        "PosteriorStats": "mat",
    },
    # per-block RMS normalization (nnet3/nnet-normalize-component.cc):
    #   p = ||x_block||^2 / (BlockDim * TargetRms^2)   (floored at 2^-66)
    #   y_block = x_block / sqrt(p); AddLogStddev appends log(rms(x_block))
    "NormalizeComponent": {
        "InputDim": "num",
        "BlockDim": "num",
        "TargetRms": "num",
        "AddLogStddev": "bool",
    },
    # LSTM cell nonlinearity (nnet3/nnet-combined-component.cc;
    # cudamatrix/cu-math.cc CpuComputeLstmNonlinearity).  Per row the
    # input is [i, f, c, o, c_prev] (5 * CellDim), the output [c_t, m_t]
    # (2 * CellDim):
    #   i_t = sigmoid(i + w_ic*c_prev);  f_t = sigmoid(f + w_fc*c_prev)
    #   c_t = f_t*c_prev + i_t*tanh(c);  o_t = sigmoid(o + w_oc*c_t)
    #   m_t = o_t*tanh(c_t)
    # Params is [3, CellDim] = (w_ic, w_fc, w_oc).  UseDropout (rows
    # carrying per-row dropout scales in 3 extra columns) is not
    # supported at inference.
    "LstmNonlinearityComponent": {
        "Params": "mat",
        "SelfRepairConfig": "ignore",
        "SelfRepairProb": "ignore",
        "UseDropout": "bool",
    },
    # truncated-BPTT gradient utility (nnet3/nnet-general-component.cc):
    # Propagate is out = Scale * in (Scale defaults to 1.0); the
    # clipping/zeroing of the RecurrenceInterval machinery only affects
    # training-time backprop
    "BackpropTruncationComponent": {
        "Dim": "num",
        "Scale": "num",
        "ClippingThreshold": "ignore",
        "ZeroingThreshold": "ignore",
        "ZeroingInterval": "ignore",
        "RecurrenceInterval": "ignore",
        "NumElementsClipped": "ignore",
        "NumElementsZeroed": "ignore",
        "NumElementsProcessed": "ignore",
        "NumZeroingBoundaries": "ignore",
        "Zeroing": "ignore",
    },
}


def _build_schema(semantic: dict) -> dict:
    """Merge the shared training-only tags into a component's schema."""
    schema = dict(TRAINING_ONLY_TAGS)
    schema.update(semantic)
    return schema


COMPONENT_SCHEMAS = {t: _build_schema(s) for t, s in _SEMANTIC_SCHEMAS.items()}


# ---------------------------------------------------------------------------
# Forward-pass classification
# ---------------------------------------------------------------------------

# Component types that are identity at inference (training-time noise /
# augmentation).  Their parameters are parsed for completeness but the
# forward pass passes features through unchanged.
IDENTITY_TYPES = {
    "NoOpComponent",
    "GeneralDropoutComponent",
    "SpecAugmentTimeMaskComponent",
}


# The network output the decoder consumes.  Chain models also carry other
# output nodes (e.g. "output-xent", a training-only regularizer branch) that
# may come anywhere in the config; inference must use this one by NAME, as
# kaldi's nnet3-compute and the decodables do -- never "the first output
# node".
OUTPUT_NODE = "output"


def component_out_dim(c: Component) -> int:
    """Output dim of one component, from its parsed params (inference)."""
    p = c.params
    if c.type in (
        "FixedAffineComponent",
        "NaturalGradientAffineComponent",
        "TdnnComponent",
    ):
        return int(p["LinearParams"].shape[0])
    if c.type == "LinearComponent":
        return int(p["Params"].shape[0])
    if c.type == "PermuteComponent":
        return len(p["ColumnMap"])
    if c.type == "BatchNormComponent":
        return int(p["Dim"])
    if c.type == "NormalizeComponent":
        d, bd = int(p["InputDim"]), int(p.get("BlockDim", p["InputDim"]))
        return d + (d // bd if p.get("AddLogStddev") else 0)
    if c.type == "RestrictedAttentionComponent":
        cd = int(p["NumLeftInputs"]) + 1 + int(p["NumRightInputs"])
        out = int(p["ValueDim"]) + (cd if p.get("OutputContext", True) else 0)
        return int(p["NumHeads"]) * out
    if c.type == "LstmNonlinearityComponent":
        return 2 * int(p["Params"].shape[1])
    if c.type == "BackpropTruncationComponent":
        return int(p["Dim"])
    if "Dim" in p:  # identity-family (NoOp, dropout, relu, ...)
        return int(p["Dim"])
    raise ValueError(f"cannot determine the output dim of {c.type!r} ({c.name})")


def output_node_index(nodes: List[Node]) -> int:
    """Index of the output-node named OUTPUT_NODE in `nodes`.  Every node
    the output depends on comes before it (kaldi writes the config
    topologically sorted), so nodes[: index + 1] is all inference needs."""
    for i, n in enumerate(nodes):
        if n.kind == "output" and n.name == OUTPUT_NODE:
            return i
    raise ValueError(f"no output-node named {OUTPUT_NODE!r} in the network")
