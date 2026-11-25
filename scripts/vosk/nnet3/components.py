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
}

# large-en: 2-D convolution over (time x height), zero-padded at both
# edges (see the torch_model.TimeHeightConvolution docstring for the
# verified semantics) and a plain column permutation.
_UNIMPLEMENTED_TYPES = set()


def _build_schema(semantic: dict) -> dict:
    """Merge the shared training-only tags into a component's schema."""
    schema = dict(TRAINING_ONLY_TAGS)
    schema.update(semantic)
    return schema


COMPONENT_SCHEMAS = {t: _build_schema(s) for t, s in _SEMANTIC_SCHEMAS.items()}
for _t in _UNIMPLEMENTED_TYPES:
    COMPONENT_SCHEMAS[_t] = None  # None = recognized, parser not implemented


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


def component_out_dim(c: Component) -> int:
    """Output dimension of a component, from its parameters."""
    if c.type == "TdnnComponent":
        return c.params["LinearParams"].shape[0]
    if c.type in ("NaturalGradientAffineComponent", "FixedAffineComponent"):
        return c.params["LinearParams"].shape[0]
    if c.type == "LinearComponent":
        return c.params["Params"].shape[0]
    if "Dim" in c.params:
        return int(c.params["Dim"])
    raise ValueError(f"cannot determine output dim of {c.type} ({c.name})")
