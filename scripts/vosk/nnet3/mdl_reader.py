# sherpa-onnx/scripts/vosk/nnet3/mdl_reader.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Reads a kaldi `am/final.mdl` acoustic model into python structures.
#
# Two things come out of a final.mdl:
#
#   1. the *TransitionModel* — the HMM half (trans-id <-> pdf-id <-> phone,
#      self-loops, silence phones).  The runtime needs it, so we split it out
#      of the binary file as a standalone kaldi object (see
#      split_transition_model_bytes); it is NOT converted to ONNX.
#
#   2. the *AmNnetSimple* network — the neural net half, which IS the thing we
#      convert to PyTorch/ONNX.  We read it from the *text* form produced by
#        nnet3-am-copy --binary=false am/final.mdl -
#      and parse that text.
#
# The text layout (verified against real vosk models) is:
#
#   <TransitionModel> ... </TransitionModel>          <- ignored here (we take
#   <Nnet3>                                               the binary sidecar)
#   input-node name=input dim=40                    <- config section: node
#   component-node name=idct component=idct ...        graph, plain text lines
#   dim-range-node name=x input-node=y dim-offset=0 dim=40
#   <NumComponents> 91
#   <ComponentName> idct <FixedAffineComponent>     <- component section:
#     <LinearParams> [ ...rows... ]                    token blocks with
#     <BiasParams> [ ... ]                             parameters
#   </FixedAffineComponent>
#   ...
#   </Nnet3> <LeftContext> 26 <RightContext> 14 <Priors> [ ]
#
# Design principles:
#   * Loud failures: unknown component types, unknown parameter tags and
#     unknown descriptor functions all raise.  A model we do not understand
#     fails the export instead of silently producing a wrong network.
#   * Streaming: parse from the stdout pipe so multi-hundred-MB text dumps of
#     the large models never touch disk.
#   * Training-only parameters (MaxChange, natural-gradient state, dropout
#     proportions, ...) are recognized, consumed and dropped.
#
# See notes.md in this directory for the full design write-up.

import re
import subprocess
from typing import List, Optional

import numpy as np

# Relative import: this file is part of the nnet3/ package.
from .components import COMPONENT_SCHEMAS, Component, Node, Nnet3Spec


# ---------------------------------------------------------------------------
# TransitionModel extraction (binary sidecar)
# ---------------------------------------------------------------------------


def split_transition_model_bytes(mdl_path: str) -> bytes:
    """Return the leading TransitionModel object of a kaldi .mdl, verbatim.

    kaldi's .mdl binary stream is a sequence of kaldi objects; the first is
    the TransitionModel.  Its serialization ends at the literal token
    ``</TransitionModel>`` (kaldi's WriteToken writes tokens verbatim), so the
    sidecar is simply the file prefix up to and including that token and the
    trailing space kaldi writes after every token.

    The returned bytes start with kaldi's ``\\0B`` binary magic and can be read
    back with ``kaldiio::Input`` + ``TransitionModel::Read`` (that is exactly
    what kaldi-hmm-gmm's python ``TransitionModel`` expects).
    """
    with open(mdl_path, "rb") as f:
        data = f.read()
    end = data.find(b"</TransitionModel>")
    if end < 0:
        raise ValueError(f"no <TransitionModel> in {mdl_path}")
    # +1 for the trailing space of WriteToken.
    return data[: end + len(b"</TransitionModel>") + 1]


# ---------------------------------------------------------------------------
# Text tokenizer
# ---------------------------------------------------------------------------


class _Tokenizer:
    """Whitespace-separated token reader over a line-oriented text stream.

    Only what the dump needs: ``next_token`` / ``peek_token`` for scalars and
    tags, plus line tracking so matrix rows can be told apart (kaldi writes
    one matrix row per line, vectors on a single line).
    """

    def __init__(self, stream):
        self.stream = stream
        self.buf = ""  # current line being tokenized
        self.pos = 0  # byte offset into buf

    def _fill(self) -> bool:
        """Advance to the next line if the current one is consumed."""
        if self.pos < len(self.buf):
            return True
        line = self.stream.readline()
        if not line:
            return False
        self.buf = line
        self.pos = 0
        return True

    def next_token(self) -> Optional[str]:
        # skip whitespace; note lines may end mid-look (that is fine: the
        # next token simply comes from the following line)
        while True:
            if not self._fill():
                return None
            while self.pos < len(self.buf) and self.buf[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buf):
                break
        start = self.pos
        while self.pos < len(self.buf) and not self.buf[self.pos].isspace():
            self.pos += 1
        return self.buf[start : self.pos]

    def peek_token(self) -> Optional[str]:
        """Look at the next token without consuming it."""
        saved = (self.buf, self.pos)
        tok = self.next_token()
        self.buf, self.pos = saved
        return tok

    def at_line_end(self) -> bool:
        """True if only whitespace remains on the current line.

        Matrix rows in kaldi's text format end at the line end, so this is
        how the matrix reader knows one row ended.
        """
        return self.pos >= len(self.buf) or self.buf[self.pos :].strip() == ""

    def read_matrix_rows(self) -> List[List[float]]:
        """Read a text matrix after its opening '[' was consumed.

        Consumes numbers until the closing ']', starting a new row at every
        line boundary (kaldi's Matrix::Write text layout).  A vector (single
        line) therefore parses as a 1-row matrix.
        """
        rows: List[List[float]] = []
        cur: List[float] = []
        while True:
            t = self.next_token()
            if t is None:
                raise ValueError("unterminated matrix in dump")
            if t == "]":
                if cur:
                    rows.append(cur)
                return rows
            if t == "[":
                raise ValueError("unexpected nested '[' in matrix")
            # kaldi writes some int lists as comma-separated pairs
            # (e.g. THConv <Offsets>: "-1,-1 -1,0 ...")
            for piece in t.split(","):
                if piece:
                    cur.append(float(piece))
            if self.at_line_end():
                rows.append(cur)  # end of a matrix row
                cur = []


# ---------------------------------------------------------------------------
# Input descriptor expressions (the node-graph mini language)
# ---------------------------------------------------------------------------
#
# component-node ... input=<expr> where <expr> is one of:
#   node-name
#   Append(a, b, ...)          concatenate on the feature axis
#   Sum(a, b)                  elementwise sum
#   Scale(c, x)                scalar multiply
#   ReplaceIndex(x, t, i)      use time-index t of x for every output frame
#                              (kaldi: fixes one index; used for the i-vector
#                              so it is constant over time)
# Anything else fails loudly.


def _split_args(s: str) -> List[str]:
    """Split a descriptor argument list on top-level commas."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def parse_descriptor(desc: str) -> tuple:
    """Parse an input descriptor into a nested tuple tree.

    ('name', s) | ('append', [..]) | ('sum', [..])
    | ('scale', float, expr) | ('offset', expr, int)
    | ('replace_index', expr, index_name, value)
    """
    desc = desc.strip()
    if re.match(r"^[A-Za-z_][A-Za-z0-9_.\-]*$", desc):
        return ("name", desc)
    m = re.match(r"^Append\((.*)\)$", desc, re.S)
    if m:
        return ("append", [parse_descriptor(a) for a in _split_args(m.group(1))])
    m = re.match(r"^Sum\((.*)\)$", desc, re.S)
    if m:
        return ("sum", [parse_descriptor(a) for a in _split_args(m.group(1))])
    m = re.match(r"^Scale\((.*)\)$", desc, re.S)
    if m:
        args = _split_args(m.group(1))
        if len(args) != 2:
            raise ValueError(f"Scale expects 2 args: {desc!r}")
        return ("scale", float(args[0]), parse_descriptor(args[1]))
    m = re.match(r"^ReplaceIndex\((.*)\)$", desc, re.S)
    if m:
        # ReplaceIndex(x, 't', v): fix the time index to the constant v, i.e.
        # use x at time v at every output frame (kaldi's descriptor language;
        # seen as ReplaceIndex(ivector, t, 0)).
        args = _split_args(m.group(1))
        if len(args) != 3:
            raise ValueError(f"ReplaceIndex expects 3 args: {desc!r}")
        return ("replace_index", parse_descriptor(args[0]), args[1], int(args[2]))
    m = re.match(r"^Offset\((.*)\)$", desc, re.S)
    if m:
        # Offset(x, t): time shift -- output frame t reads x at t + offset,
        # clamped at utterance edges (same convention as Tdnn offsets).
        args = _split_args(m.group(1))
        if len(args) != 2:
            raise ValueError(f"Offset expects 2 args: {desc!r}")
        return ("offset", parse_descriptor(args[0]), int(args[1]))
    raise ValueError(f"unsupported descriptor: {desc!r}")


# ---------------------------------------------------------------------------
# Config section (node graph)
# ---------------------------------------------------------------------------


def _parse_config_line(line: str, nodes: List[Node]) -> None:
    """Parse one config line into a Node.

    Three line kinds (all seen in vosk models):
      input-node name=X dim=D
      component-node name=X component=C input=<expr>
      dim-range-node name=X input-node=Y dim-offset=O dim=K
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return
    if line.startswith("input-node "):
        name = re.search(r"name=(\S+)", line).group(1)
        dim = int(re.search(r"dim=(\S+)", line).group(1))
        nodes.append(Node(name=name, kind="input", dim=dim))
    elif line.startswith("component-node "):
        name = re.search(r"name=(\S+)", line).group(1)
        comp = re.search(r"component=(\S+)", line).group(1)
        # the descriptor is everything after the first 'input='
        desc = line.split("input=", 1)[1].strip()
        nodes.append(Node(name=name, kind="component", component=comp, descriptor=desc))
    elif line.startswith("output-node "):
        # output-node name=X input=<expr> [objective=linear|chain|xent]
        # The objective= suffix is training-related; drop it before parsing
        # the input descriptor.
        name = re.search(r"name=(\S+)", line).group(1)
        desc = line.split("input=", 1)[1]
        desc = re.sub(r"\s+objective=\S+", "", desc).strip()
        nodes.append(Node(name=name, kind="output", descriptor=desc))
    elif line.startswith("dim-range-node "):
        name = re.search(r"name=(\S+)", line).group(1)
        src = re.search(r"input-node=(\S+)", line).group(1)
        off = int(re.search(r"dim-offset=(\S+)", line).group(1))
        dim = int(re.search(r"dim=(\S+)", line).group(1))
        nodes.append(
            Node(name=name, kind="dim-range", source=src, dim_offset=off, dim=dim)
        )
    else:
        raise ValueError(f"unsupported config line: {line!r}")


# ---------------------------------------------------------------------------
# Component section
# ---------------------------------------------------------------------------


def _read_component_params(tok: _Tokenizer, typ: str, schema: dict, name: str) -> dict:
    """Read one component's parameters until </Type>.

    Each parameter is either a scalar (number / T / F) or a '['-delimited
    matrix/vector.  ``schema`` maps tag -> kind ('mat' | 'vec' | 'offsets' |
    'num' | 'bool' | 'ignore'); unknown tags raise.
    """
    params = {}
    while True:
        t = tok.next_token()
        if t is None:
            raise ValueError(f"unterminated component {name!r}")
        if t == f"</{typ}>":
            return params
        if t.startswith("</"):
            continue  # closing tag of a nested struct (e.g. </ConvolutionModel>)
        if not (t.startswith("<") and t.endswith(">")):
            raise ValueError(f"bad parameter tag {t!r} in component {name!r}")
        tag = t[1:-1]
        if tag not in schema:
            raise ValueError(f"unknown parameter <{tag}> in {typ} ({name!r})")
        kind = schema[tag]
        if kind == "struct":
            # structural opener with no value; nested tags carry the params
            continue
        if kind == "pairs":
            tok.next_token()  # '['
            rows = tok.read_matrix_rows()
            flat = [int(x) for x in rows[0]] if rows else []
            if len(flat) % 2 != 0:
                raise ValueError(f"<{tag}> expects pairs: {name!r}")
            params[tag] = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
            continue
        if kind == "flag":
            # kaldi writes presence flags with no value (tag alone); there is
            # nothing to consume.
            params[tag] = True
            continue
        if tok.peek_token() == "[":
            tok.next_token()  # consume '['
            rows = tok.read_matrix_rows()
            if kind == "mat":
                if not rows:
                    # an empty matrix (e.g. no bias): 0x0
                    params[tag] = np.zeros((0, 0), dtype=np.float32)
                else:
                    params[tag] = np.array(rows, dtype=np.float32)
            elif kind == "vec":
                if not rows:
                    # '[ ]' = absent vector (e.g. TdnnComponent with no bias)
                    params[tag] = np.zeros(0, dtype=np.float32)
                elif len(rows) != 1:
                    raise ValueError(
                        f"<{tag}> should be a vector, "
                        f"got {len(rows)} rows ({name!r})"
                    )
                else:
                    params[tag] = np.array(rows[0], dtype=np.float32)
            elif kind == "offsets":
                if len(rows) != 1:
                    raise ValueError(f"<{tag}> should be a single row " f"({name!r})")
                params[tag] = [int(x) for x in rows[0]]
            # 'ignore' with matrix payload: consumed and dropped
        elif kind == "ignore":
            # Consume every value up to the next tag: training-only fields
            # have arbitrary arity (e.g. <AlphaInOut> 4 4 is a pair).
            while True:
                v = tok.peek_token()
                if v is None or v.startswith("<"):
                    break
                tok.next_token()
        else:
            v = tok.next_token()
            if v is None or v.startswith("<"):
                # never eat a tag/closing tag as a scalar value
                raise ValueError(
                    f"missing value for <{tag}> in {name!r}, " f"got {v!r}"
                )
            if kind == "num":
                params[tag] = float(v)
            elif kind == "bool":
                params[tag] = v == "T"


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def parse_text_dump(stream) -> Nnet3Spec:
    """Parse the text form of an AmNnetSimple (nnet3-am-copy --binary=false).

    ``stream`` is a text file-like object, typically the stdout pipe of
    nnet3-am-copy (streaming; see the module docstring).
    """
    tok = _Tokenizer(stream)

    # Skip everything before <Nnet3> (the binary-only TransitionModel
    # section of the text dump; the sidecar comes from the binary file).
    while True:
        t = tok.next_token()
        if t is None:
            raise ValueError("no <Nnet3> in dump")
        if t == "<Nnet3>":
            break

    # Config section: plain lines up to <NumComponents>.
    nodes: List[Node] = []
    num_components = None
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("no <NumComponents> in dump")
        if line.startswith("<NumComponents>"):
            num_components = int(line.split()[1])
            tok.buf, tok.pos = "", 0  # reset tokenizer onto the new section
            break
        _parse_config_line(line, nodes)

    # Component section: repeated <ComponentName> name <Type> ... </Type>.
    components: List[Component] = []
    while True:
        t = tok.next_token()
        if t == "</Nnet3>":
            break
        if t != "<ComponentName>":
            raise ValueError(f"expected <ComponentName>, got {t!r}")
        name = tok.next_token()
        typ = tok.next_token()
        if not (typ.startswith("<") and typ.endswith(">")):
            raise ValueError(f"bad component type token: {typ!r}")
        typ = typ[1:-1]
        # Unknown types raise; recognized-but-unimplemented (large-en's
        # TimeHeightConvolution / Permute) raise NotImplementedError until
        # their parsing is added.
        if typ not in COMPONENT_SCHEMAS:
            raise ValueError(f"unknown component type {typ!r} ({name!r})")
        schema = COMPONENT_SCHEMAS[typ]
        if schema is None:
            raise NotImplementedError(
                f"component type {typ!r} ({name!r}): parsing not implemented "
                f"yet (large-en model; see)"
            )
        params = _read_component_params(tok, typ, schema, name)
        components.append(Component(name=name, type=typ, params=params))

    if num_components is not None and len(components) != num_components:
        raise ValueError(
            f"parsed {len(components)} components, " f"expected {num_components}"
        )

    # Tail: <LeftContext> N <RightContext> N <Priors> [ ... ]
    # (the receptive field — useful as a sanity check for meta.json)
    left = right = 0
    while True:
        t = tok.next_token()
        if t is None:
            break
        if t == "<LeftContext>":
            left = int(tok.next_token())
        elif t == "<RightContext>":
            right = int(tok.next_token())
        elif t == "<Priors>":
            break  # the priors vector itself is not needed (chain models)

    # Determine the I/O dims from the graph.
    input_node = next((n for n in nodes if n.name == "input"), None)
    ivector_node = next((n for n in nodes if n.name == "ivector"), None)
    output_node = next((n for n in nodes if n.name == "output"), None)
    if input_node is None or output_node is None:
        raise ValueError("dump must have 'input' and 'output' nodes")
    # 'output' is an output-node: its descriptor names the feeding node (e.g.
    # input=output.affine).  Resolve dims through that node's component.
    out_desc = parse_descriptor(output_node.descriptor)
    if out_desc[0] != "name":
        raise ValueError(
            f"output descriptor must be a plain node name, "
            f"got {output_node.descriptor!r}"
        )
    feed = next(n for n in nodes if n.name == out_desc[1])
    if feed.kind != "component":
        raise ValueError(f"output feeds non-component node {feed.name!r}")
    out_comp = next(c for c in components if c.name == feed.component)
    if "Dim" in out_comp.params:
        output_dim = int(out_comp.params["Dim"])
    elif "LinearParams" in out_comp.params:
        output_dim = out_comp.params["LinearParams"].shape[0]
    elif "Params" in out_comp.params:
        output_dim = out_comp.params["Params"].shape[0]
    else:
        raise ValueError(f"cannot determine output dim of {out_comp.type}")

    return Nnet3Spec(
        components=components,
        nodes=nodes,
        input_dim=input_node.dim,
        ivector_dim=ivector_node.dim if ivector_node else 0,
        output_dim=output_dim,
        left_context=left,
        right_context=right,
    )


# ---------------------------------------------------------------------------
# Convenience: run nnet3-am-copy and parse its output
# ---------------------------------------------------------------------------


def dump_via_kaldi(mdl_path: str, nnet3_am_copy: str = "nnet3-am-copy") -> Nnet3Spec:
    """Parse ``mdl_path`` by streaming ``nnet3-am-copy --binary=false ... -``.

    This is the v1 entry point of: kaldi is used only
    as a dumper (locally or in the ghcr.io/csukuangfj/kaldi container); the dump
    itself is parsed here and never lands on disk.
    """
    proc = subprocess.Popen(
        [nnet3_am_copy, "--binary=false", mdl_path, "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        # text=True already wraps the pipe in a text stream; do not wrap again
        spec = parse_text_dump(proc.stdout)
        return spec
    finally:
        # Never wait() on a producer whose pipe we stopped reading: with a
        # multi-MB dump it blocks forever on a full pipe.  Kill and drain.
        proc.kill()
        try:
            proc.stdout.read()
        except Exception:
            pass
        proc.wait()
