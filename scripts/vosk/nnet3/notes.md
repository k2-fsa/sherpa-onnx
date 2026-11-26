# nnet3/ — design notes

How we read a kaldi `am/final.mdl` (the Vosk acoustic model) for the
PyTorch/ONNX export pipeline (`scripts/vosk/`), and why each piece is
built the way it is.  Self-contained reference for the nnet3-reading
part of the export; usage lives in the top-level `README.md`.

## 0. The moving parts

```
final.mdl ──nnet3-am-copy --binary=false──> text dump ──mdl_reader.py──> Nnet3Spec
                                                                          │
        ┌─────────────────────────────────────────────────────────────────┘
        ▼
  components.py     per-component parameter schemas (what each <Tag> means)
  torch_model.py    one nn.Module per component + Nnet3Torch.forward
  export_torch.py   Nnet3Spec -> am/am.pt (+ am/am_flat.py, model_meta.json)
  export_onnx.py    am.pt -> am/final.onnx (+ final.int8.onnx)
  flatten.py        Nnet3Spec -> readable explicit-wiring module (am_flat.py)
```

Everything downstream consumes one dataclass, `Nnet3Spec`: a list of
`Node`s (the graph) and a list of `Component`s (typed parameter blobs),
plus the dims and the receptive field.

## 1. What a final.mdl is

A kaldi `.mdl` binary stream holds two concatenated kaldi objects:

```
final.mdl = [TransitionModel object][AmNnetSimple object]
```

* **TransitionModel** — the HMM half: trans-id ↔ pdf-id ↔ phone mapping,
  topology, self-loops, silence phones.  The *runtime* needs it (decoding,
  endpointing, timestamps); it is NOT part of the ONNX conversion.  We split
  it out verbatim as `am/transition_model.mdl` (see "TransitionModel
  sidecar" below).
* **AmNnetSimple** — the neural net: a node graph + a list of components with
  parameters.  This is what becomes the PyTorch/ONNX model.

## 2. How the network is read

The network is parsed from the **text** form produced by one kaldi call:

```
nnet3-am-copy --binary=false am/final.mdl -
```

The dump is consumed as a **stdout pipe** (`dump_via_kaldi()`); it is never
written to disk.  Reason: text is ~3x the binary (measured: small-cn
15.2 MB → 44.8 MB; large-cn would be ~280 MB), and streaming bounds memory
to one component's parameters.  Kaldi is only a *dumper* — any kaldi build
works (point `KALDI_ROOT` at it); the parser itself is pure python.

(A possible future alternative is parsing the binary form directly, which
would drop even the dumper tool; the text form is easier to debug and the
format quirks below are now well understood.)

### Text dump layout (verified against all five supported models)

```
<TransitionModel> ... </TransitionModel>        <- ignored (binary sidecar instead)
<Nnet3>
input-node name=ivector dim=40                 <- config section: plain text
input-node name=input dim=40                      lines describing the node
component-node name=idct component=idct input=input
dim-range-node name=x input-node=y dim-offset=0 dim=40
output-node name=output input=output.affine objective=linear
<NumComponents> 91
<ComponentName> idct <FixedAffineComponent>    <- component section: token
  <LinearParams> [                              blocks with parameters
   ... one row per line ...
  ]
  <BiasParams> [ 0 0 ... ]
</FixedAffineComponent>
...
</Nnet3> <LeftContext> 26 <RightContext> 14 <Priors> [ ]
```

Four line kinds in the config (`input-node`, `component-node`,
`dim-range-node`, `output-node`), token blocks in the component section, and
a tail carrying the **receptive field** (`LeftContext`/`RightContext` — kept
as a sanity check for `model_meta.json`).

## 3. TransitionModel sidecar (`split_transition_model_bytes`)

kaldi's binary serialization writes tokens verbatim
(`WriteToken` = token + one space), so the TransitionModel object is exactly
the file prefix ending at the literal `</TransitionModel> ` token.  The
returned bytes keep the `\0B` binary magic and can be read directly by
`kaldiio::Input` + `TransitionModel::Read` — i.e. by the `kaldi_hmm_gmm`
python package at runtime.

## 4. Parsing design (mdl_reader.py)

* **`_Tokenizer`** — minimal whitespace token reader over the line-oriented
  stream.  It tracks line boundaries because kaldi writes **one matrix row
  per line** (vectors on a single line), which is how rows are told apart.
* **`_read_component_params`** — walks `<Tag> value ...` until `</Type>`,
  driven by the per-type schema in `components.py`.
* **`parse_text_dump`** — config lines → `Node`s, component blocks →
  `Component`s, tail → receptive field, then I/O dims are resolved from the
  graph (`input` node, `ivector` node, and the component feeding `output`).

### Format quirks discovered (each one initially broke the parser)

1. **Valueless presence flags**: `<Continuous> </GeneralDropoutComponent>`
   — kaldi writes some booleans as a bare tag with *no* value.  Schema kind
   `'flag'` handles these (absence = False).
2. **Pair-valued tags**: `<AlphaInOut> 4 4`, `<RankInOut> 20 48` — some
   training-only fields carry two numbers.  The `'ignore'` kind therefore
   consumes *all* values up to the next tag, regardless of arity.
3. **Empty parameters**: `<BiasParams> [ ]` means "no bias" (kaldi's
   `kPropagateAdds` case for TdnnComponent).  Empty matrices/vectors parse
   as zero-size arrays; the forward pass then simply adds no bias.
4. **`output-node` lines** in the config have an `objective=...` suffix which
   must be stripped before parsing the input descriptor.  Chain models
   have two of them: `output` (what the decoder consumes) and the
   training-only `output-xent` branch.  Inference selects the node named
   `output` and evaluates only the nodes it depends on
   (`mdl_reader.inference_nodes`), whatever the order of the two in the
   config -- never "the first output node".
5. **Pipe deadlock**: `subprocess.wait()` while the producer still has
   megabytes of un-consumed output blocks forever.  `dump_via_kaldi` kills
   and drains the pipe in `finally`.
6. `nnet3-am-copy` is invoked with `-` (stdout) and the output wrapped in a
   text stream **once** (`text=True`); double-wrapping with `TextIOWrapper`
   breaks reads.

### Error policy

Loud, always: unknown component types, unknown parameter tags, unknown
descriptor functions and malformed blocks all raise.  A model we do not
understand must fail the export — never silently produce a wrong net.
All component types used by the supported models are handled, including
`TimeHeightConvolutionComponent` and `PermuteComponent` (large-en).

## 5. Node-graph mini-language

`component-node ... input=<expr>` descriptors:

| expr | meaning | real use |
|---|---|---|
| `name` | output of node `name` | most edges |
| `Append(a, b, ...)` | concatenate on the feature axis | joining the i-vector |
| `Sum(a, b)` | elementwise sum | TDNNF residual connections |
| `Scale(c, x)` | scalar multiply | `Scale(0.75, x)` in residuals |
| `Offset(x, t)` | time shift: `out[t] = x[t+t']` for shift t', clamped at edges | the delta (velocity/acceleration) layer |
| `ReplaceIndex(x, t, i)` | use time-index `i` of `x` at every frame | i-vector: `ReplaceIndex(ivector, t, 0)`.  kaldi's online decoders rewrite it to `Round(ivector, chunk)` (one i-vector per chunk); the exported model's ivector input is per frame and this reads row `t` (`torch_model.is_ivector_index`) |
| `IfDefined(x)` | `x` where it is computable at that row, else **zeros** (see §7b) | LSTM feedback: `IfDefined(Offset(lstm1.c_trunc, -3))` |
| `Failover(a, b)` | `a` where computable, else `b` | (grammar completeness) |
| `Const(v, dim)` | constant row | (grammar completeness) |
| `Round(x, n)` | read row `(t // n) * n` of `x` | (grammar completeness) |
| `Switch(a, b, ...)` | read option `t % n` | (grammar completeness) |
| `dim-range-node` | named node slicing dims `[dim_offset, dim_offset+dim)` of a source node | pulling slices of a wider node |

`parse_descriptor` validates and converts these to a nested tuple tree that
`torch_model.py` evaluates.  Anything else raises.

Real descriptor lines from small-cn (they read cryptic until you know the
table):

```
# the i-vector joins the delta features, fixed at time 0:
component-node name=input2 component=input2
  input=Append(delta, ReplaceIndex(ivector, t, 0))

# a TDNNF block adds a scaled residual of the block's input:
component-node name=tdnnf2.noop component=tdnnf2.noop
  input=Sum(Scale(0.75, tdnn1.dropout), tdnnf2.dropout)
```

## 6. Component parameter schemas (components.py)

Schemas map parameter tags to kinds (`mat` / `vec` / `offsets` / `num` /
`bool` / `flag` / `ignore`).  Two groups:

* **semantic** parameters (weights, offsets, batchnorm stats) — typed and
  validated;
* **training-only** tags (`TRAINING_ONLY_TAGS`: natural-gradient state,
  running objective averages, self-repair counters, ...) — accepted for every
  component and dropped.

Per-type inference semantics are documented in the `components.py` header
(e.g. `TdnnComponent` weight layout `[out, in × num_offsets]` column-blocked
per offset — from `nnet-convolutional-component.h`; `BatchNormComponent`
test-mode formula — from `nnet-normalize-component.cc`).

### Inference-time identity components

Three component types are **no-ops at inference** and map to
`torch_model.Identity` (`IDENTITY_TYPES`):

* `NoOpComponent` — pass-through by definition;
* `GeneralDropoutComponent` — dropout is a training regularizer; kaldi's
  test mode is identity (validated by the kaldi parity gates below);
* `SpecAugmentTimeMaskComponent` — training-time data augmentation; identity
  in test mode.

They stay **in the graph** (don't prune them): downstream descriptors
reference their node names (the TDNNF residual above reads `tdnn1.dropout`),
and keeping them keeps the graph 1:1 with kaldi's.  At runtime they cost
nothing (ORT eliminates `Identity` nodes).

## 7. Edge semantics (important — verified experimentally)

The network looks at neighbouring frames (Tdnn offsets, `Offset()`), so at the
utterance boundaries it needs frames outside `[0, T)`.  Two natural policies
exist:

* **per-layer clamping** — each gather clamps its index to `[0, T-1]`
  (replicating the *previous layer's* edge output);
* **replicated input extension** — extend the *input* features by
  `LeftContext`/`RightContext` replicated edge frames, run the whole network
  (intermediate layers then compute real values at out-of-range times), and
  crop the output rows back to `[0, T)`.

They are NOT equivalent (nonlinear layers make replicate-at-input differ from
replicate-per-layer), and kaldi uses the second one: measured against
`nnet3-compute` on small-cn, per-layer clamping differs by up to 8.2 at the
first/last frames, while input extension + crop matches to **1.7e-5
everywhere**.  So the caller extends the input: `torch_model.Nnet3Torch.forward`
is pure (it computes exactly the rows it is given), and the recognizer's
windows carry `LeftContext`/`RightContext` frames of real history or
replicated edge frames, keeping only the inner rows; the per-layer clamps
remain only as a safety net.  (This is also consistent with
how kaldi's looped decodable assembles chunks: it feeds clamped *raw* context
rows and lets the computation produce true intermediate values.)

### Worked example (L=2, R=1, single window)

For an output row at frame `t` the net needs input frames `[t-2, t+1]`.
On a 5-frame utterance `x0..x4`, replicated input extension gives

```
input : [x0 x0]  x0  x1  x2  x3  x4 [x4]      <- replicas only at the edges
y0 = f(x0, x0, x0, x1)     y3 = f(x1, x2, x3, x4)
y1 = f(x0, x0, x1, x2)     y4 = f(x2, x3, x4, x4)
y2 = f(x0, x1, x2, x3)     ← the middle rows never touch a replica
```

Per-layer clamping would instead clamp *every* layer's gathers, so the
replication happens again and again as the signal flows deeper — the source
of the 8.2 discrepancy.  The same picture drives **chunked/streaming**
evaluation: feed the window `[t0-L, t1+R)` and keep only rows `[t0, t1)`;
their receptive fields sit strictly inside the window, so the result is
identical to the one-shot computation (see README, "Notes on semantics").

## 7b. Recurrent nets (LSTM) — verified on gigaspeech

`vosk-model-en-us-0.42-gigaspeech` is the only recurrent model among the
32: two clockwork LSTM blocks (`lstm1`, `lstm2`), each wired as a cycle in
the node graph,

```
lstm1.W_all      = affine(Append(x, IfDefined(Offset(lstm1.r_trunc, -3))))
lstm1.lstm_nonlin = LstmNonlinearity(Append(lstm1.W_all,
                                            IfDefined(Offset(lstm1.c_trunc, -3))))
lstm1.cr_trunc   = BackpropTruncation(Append(lstm1.c, lstm1.r))   # c/r fed back
```

i.e. three interleaved recurrences (rows `t ≡ 0, 1, 2 mod 3`), each
stepping 30 ms.  `Nnet3Torch` evaluates such cycles row by row
(`_eval_recurrent`); the flat module emits one TorchScript loop module per
cycle (`_Rec<i>`), which ONNX export keeps as a `Loop`.

**Where the loop starts (the zero state).**  kaldi never invents rows: an
input row exists only if it was fed, intermediate nodes do not clamp, and
a row is *computable* only if everything it reads is.  `IfDefined` adds no
constraint (it reads zeros where its argument is not computable), which
is what breaks the cycle.  So a loop's zero-state rows are exactly the
rows below the first computable row of the loop's non-recurrent input --
**each cycle has its own origin**, deeper cycles start later.  With the
input padded by `left_context` (84) rows, gigaspeech's lstm1 starts at
fed row 81 and lstm2 at fed row 84.  `torch_model.first_computable_rows`
computes this statically (descriptor offsets + each component's own time
offsets); neither "fed row 0" nor "first real frame" is right in general.

**Reference pitfall: `nnet3-compute` chunks.**  Its default
`--frames-per-chunk=50` computes each chunk independently and so restarts
every recurrence each 50 frames: kaldi vs kaldi (50 vs one chunk) differs
by 9.43, starting exactly at row 50.  The gates therefore run
`nnet3-compute --frames-per-chunk=1000000` (identical results for
non-recurrent nets).  A continuous forward pass -- and vosk's looped
online decodable, which carries the state -- correspond to the
one-chunk output.

**Float-noise amplification.**  Layer by layer (a kaldi net per node with
that node as output, unchunked, vs our node with the sub-net's own
padding), every node from `idct` through lstm1 and `tdnnf18` matches to
~1e-6 relative.  Only lstm2's `t ≡ 0 (mod 3)` chain drifts after row
~250 -- and it does so in kaldi too: kaldi on the original `final.mdl` vs
kaldi on the text dump re-encoded to binary (7 significant digits)
differs by **0.42 at the same row**.  Any perturbation (rounding, BLAS
summation order, ORT vs eager kernels) is amplified there, so a 1e-3 max
gate is unreachable by *any* re-implementation, and parsing the binary
`final.mdl` would not help.  The gates for recurrent nets (`compare_torch`,
`compare_onnx`: `model_meta.json` `recurrent=1`) fall back to median
per-row max <= 1e-3 plus argmax agreement >= 0.99; measured on en.wav:
median 1.4e-5, argmax 1.000, worst 0.49 (row 360).

**Streaming: the state is an explicit input/output.**  The recognizers
run the AM on windows `[t - gl, t + n + R)` (`gl` = left context rounded
up to the row grid, `n` = new frames).  Without a state each window
would restart the LSTMs, which is badly wrong: on en.wav at chunk 30 the
rows fed to the decoder then differ from the one-shot run by up to 9.7
(argmax agreement 0.72 -- that en.wav still decoded correctly was luck).

What makes an exact state possible: a loop's origin in a window is its
first computable row, which depends only on the window START (lstm1: 81,
lstm2: 84 for `gl` = 84), and the loop at row `t` reads only the rows
`t - lag` (lag 3).  So the state is the `state_rows` = 3 rows just
before the origin of every fed-back node (`recurrent_state_layout`:
`lstm1.c_trunc, lstm1.r_trunc, lstm2.c_trunc, lstm2.r_trunc` ->
`[B, 3, 2560]`), injected in place of the blank rows; zeros = the start
of the utterance.  The next window starts `shift = n` frames later, so
its state is rows `[origin + n - 3, origin + n)` of this window -- and
those are fully computable here: they need input up to
`origin + n - 1 + rx` (rx = the loop input's right context, 57 for
lstm2), and the window reaches `gl + n + R - 1`.

* torch: `Nnet3Torch.forward_streaming(features, ivector, state, shift)
  -> (log_probs, state_out)`, `initial_state(batch)`; the generated
  `FlatAm` has the same methods (its `_Rec<i>` loops read the state).
* ONNX (recurrent models only): inputs `features, ivector, state`,
  outputs `log_probs, state_out`; `shift = T - gl - R` is derived from
  the window length inside the graph -- the recognizer's window layout
  is the contract.  Metadata: `recurrent=1`, `state_rows`, `state_dim`.
  Non-recurrent models keep the `features, ivector -> log_probs`
  interface.
* recognizer: `stream.am_state` carries the state; every AM backend has
  `run(feats, ivector, state, shift) -> (rows, state_out)`.

Verified (gigaspeech, en.wav): chaining windows of 15/30/60 frames
reproduces the one-shot rows **bit-exactly** (max diff 0) for
`Nnet3Torch`, the flat module, ONNX float32 and ONNX int8 -- both in a
direct window harness and through `OnlineRecognizer` vs
`OfflineRecognizer` (i-vector pinned; audio fed in 100 ms pieces).
`export_onnx.py` gates it: a non-blank state handover onnx-vs-torch
(4e-5) and chained windows vs one-shot (0.0).  Note that real streaming
also updates the i-vector per window (like vosk), so its rows
legitimately differ from the offline run's whole-utterance i-vector.

## 8. Subsampling lives outside the net

`nnet3-compute` (and `Nnet3Torch`) emit **one row per input frame** (full
rate).  kaldi anchors decoder row `k` at input frame `3*k` (no offset --
`nnet-compile-looped.cc:143-146`), so the model output is rows
`0::3` of the full-rate computation — applied by `torch_model.VoskAmWrapper`
at export time (inside the ONNX graph).  Keeping it outside the net keeps
the row-by-row kaldi comparison direct.

## 9. Test status

```sh
cd scripts/vosk
python3 compare_torch.py --model-dir <vosk-model-dir> --wav x.wav
```

Verified on `vosk-model-small-cn-0.22`:

* parsed dims: input 40, ivector 40, output 2112
  (= num_pdfs), 91 components, 97 nodes, LeftContext/RightContext 26/14
* component inventory matches the model's actual contents exactly
  (22 Tdnn, 18 BatchNorm, 14 ReLU, 13 NoOp, 13 GeneralDropout, 5
  NaturalGradientAffine, 3 Linear, 1 FixedAffine, 1 LogSoftmax,
  1 SpecAugmentTimeMask)
* `transition_model.mdl` sidecar: 171,184 bytes, `\0B<TransitionM...` ...
  `</TransitionModel> `

`compare_torch.py` (torch vs kaldi `nnet3-compute`, same input; the gate
before ONNX export):

| model | wav | rows | worst max_abs | cosine |
|---|---|---|---|---|
| small-cn | test.wav | 829 | 1.7e-5 | 1.000000 |
| small-cn | zh.wav | 474 | 2.2e-5 | 1.000000 |
| small-cn | en.wav | 383 | 1.8e-5 | 1.000000 |
| small-en-us | zh.wav | 474 | 1.1e-5 | 1.000000 |
| small-en-us | en.wav | 383 | 1.2e-5 | 1.000000 |

All row-for-row comparisons of the full-rate output pass (float noise only).

Recurrent: `vosk-model-en-us-0.42-gigaspeech` (en.wav, 383 rows): median
row max_abs 1.4e-5, argmax agreement 1.000, worst 0.49 (float noise
amplified by lstm2, see §7b); ONNX stage 1 median 1.5e-5, argmax 1.000.

All five supported models (small/large, cn/en/de) parse and pass the
torch-vs-kaldi and decode checks with this reader; the scripts that consume
it live next to it in `scripts/vosk/` (see the top-level `README.md`).
