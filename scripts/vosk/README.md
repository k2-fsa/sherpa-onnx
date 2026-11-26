# vosk model → ONNX export & test tools

Export [vosk](https://alphacephei.com/vosk) acoustic models (kaldi
`am/final.mdl`) to PyTorch and ONNX for use with sherpa-onnx, and verify
the results against kaldi.  The exported models keep the **original vosk
directory layout**, so a model dir remains complete and self-contained.

```
vosk model dir ──export_torch.py──► torch model dir ──export_onnx.py──► onnx model dir
      (am/final.mdl)                  (am/am.pt)                    (am/final.onnx)
```


## Contents

- [Getting started](#getting-started)
  - [1. Set up the environment](#1-set-up-the-environment)
  - [2. Example 1 — Chinese (vosk-model-small-cn-0.22)](#2-example-1--chinese-vosk-model-small-cn-022)
  - [3. Export to ONNX](#3-export-to-onnx)
  - [4. Use the ONNX model](#4-use-the-onnx-model)
  - [5. Example 2 — English (vosk-model-small-en-us-0.15)](#5-example-2--english-vosk-model-small-en-us-015)
- [Other models / batch export](#other-models--batch-export)
- [Model directory layouts](#model-directory-layouts)
- [Readable model code (am_flat.py)](#readable-model-code-am_flatpy)
- [Verifying against kaldi](#verifying-against-kaldi)
- [Notes on semantics](#notes-on-semantics)

## Getting started

### 1. Set up the environment

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx/scripts/vosk
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy onnx onnxruntime soundfile librosa \
             kaldi-native-fbank 'kaldi-hmm-gmm>=1.3.10' kaldi-native-io kaldifst \
             onnxscript 'kaldi-decoder>=0.6.4' vosk
```

`kaldi-hmm-gmm >= 1.3.10` provides kaldi's `OnlineIvectorFeature` and
`OnlineSilenceWeighting`, `kaldi-decoder >= 0.6.4` the
`LatticeFasterOnlineDecoder` -- the pieces `recognizer.py` needs to
decode exactly like the vosk runtime.  The `vosk` package is the
reference the tests compare against.

`onnxscript` is required by the Dynamo ONNX exporter that `torch.onnx.export`
uses by default in recent PyTorch (see the
[PyTorch ONNX exporter docs](https://docs.pytorch.org/docs/stable/onnx_export.html)).

You also need a [kaldi](https://github.com/kaldi-asr/kaldi) build for the
reference tools (`nnet3-am-copy`, `nnet3-compute`, `compute-mfcc-feats`) —
they are used as *references and conversion tools*, not at runtime.  Point
the scripts at it:

```bash
export KALDI_ROOT="$HOME/kaldi"     # the scripts look for $KALDI_ROOT/src
```

(Older wheels: `kaldi-decoder < 0.6.0` exports their bundled
openfst symbols, whose FST-type registry then breaks reading the vosk
graph types (`olabel_lookahead`/`ngram` in HCLr.fst/Gr.fst) when loaded
alongside kaldifst.  0.6.0 hides them.  `kaldi-hmm-gmm` (1.3.5) still
exports its copy, so `kaldifst` must be imported **before**
`kaldi_hmm_gmm` (recognizer.py does; otherwise reading HCLr.fst fails with
`Unknown FST type olabel_lookahead`).)

Non-16 kHz audio is resampled with **librosa** (deterministic — the
sherpa-onnx python-api-examples convention).

### 2. Example 1 — Chinese (vosk-model-small-cn-0.22)

#### (a) Download the model and a test wav

```bash
curl -L -o model.zip https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip -q model.zip     # -> ./vosk-model-small-cn-0.22/
curl -L -o zh.wav https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/zh.wav
```

#### (b) Export to PyTorch

```bash
python3 export_torch.py --in-dir ./vosk-model-small-cn-0.22 --out-dir ./torch-model-cn
```

Output directory structure (original layout; `am/final.mdl` is replaced by
the pieces below):

```
./torch-model-cn/
├── am/
│   ├── am.pt                 # the torch model: {"state_dict", "args"}
│   │                         #   -- weights + the parsed network; rebuild
│   │                         #   with Nnet3Torch.from_export_args(args)
│   ├── am_flat.py            # auto-generated readable nn.Module: every
│   │                         #   connection spelled out as plain python
│   │                         #   (see "Readable model code" below)
│   ├── model_meta.json       # dims, frame_subsampling_factor,
│   │                         #   left/right context (receptive field)
│   └── transition_model.mdl  # the TransitionModel half of final.mdl
│                             #   (HMM side; needed by the decoder)
├── conf/                     # copied unchanged (model.conf, mfcc.conf)
├── ivector/                  # copied unchanged (i-vector extractor)
├── graph/                    # copied unchanged (HCLr.fst + Gr.fst)
└── README                    # copied unchanged
```

#### (c) Verify the export (against kaldi)

```bash
python3 compare_torch.py --model-dir ./vosk-model-small-cn-0.22 --wav ./zh.wav \
    --am-pt ./torch-model-cn/am/am.pt
```

Expected ending (the torch model matches kaldi's `nnet3-compute`
row-by-row):

```
rows compared: 474   worst max_abs=0.000024 (row 2)   worst cosine=1.000000
Torch == KALDI: OK
```

All six decode scripts are built on `recognizer.py`, which decodes
**exactly like the vosk runtime** (vosk-api `src/recognizer.cc`, i.e.
kaldi's online2 pipeline), with a sherpa-onnx style API
(`csrc/online-recognizer.h`, `offline-recognizer.h`, `online-stream.h`):

```python
rec = OnlineRecognizer(OnnxAm(model_dir))
stream = rec.create_stream()
for chunk in chunks:
    stream.accept_waveform(sample_rate, chunk)
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    print(rec.get_result(stream))
stream.input_finished()
while rec.is_ready(stream):
    rec.decode_stream(stream)
print(rec.get_result(stream))
```

The scripts differ only in the AM backend they plug in — `OnnxAm` (the
exported `final.onnx`, `--int8` for `final.int8.onnx`), `TorchAm`
(`Nnet3Torch` over `am/am.pt`) or `FlatAm` (the generated
`am/am_flat.py`); everything else is shared:

| vosk runtime (kaldi online2) | `recognizer.py` |
|---|---|
| MFCC (`conf/mfcc.conf`) | `kaldi_native_fbank.OnlineMfcc`, dither 0 |
| `OnlineIvectorFeature`: kaldi defaults, `max_count=100`, most recent i-vector | `kaldi_hmm_gmm.OnlineIvectorFeature` |
| `OnlineSilenceWeighting`: weight 0.001, `--endpoint.silence-phones` | `kaldi_hmm_gmm.OnlineSilenceWeighting` |
| looped decodable: chunks of `--frames-per-chunk` (24) frames, one i-vector per chunk | `IvectorChunker` + the AM on each chunk's window (per-frame ivector input) |
| `SingleUtteranceNnet3IncrementalDecoder` (`conf/model.conf` beams) | `kaldi_decoder.LatticeFasterOnlineDecoder` |
| `AcceptWaveform`: 0.2 s steps; each: update silence weights, advance the decoder | `accept_waveform` queues 0.2 s steps; `decode_stream` runs one |
| `FinalResult` with `SetMaxAlternatives(1)`: best path, graph cost x 0.9 | `get_result` after `input_finished` |

So the text is the vosk package's 1-best for the same audio (with
vosk's MFCC dither off -- vosk dithers by default, which makes its own
output random).  Not reproduced: vosk's default MBR result
(`SetMaxAlternatives(0)`), the lattice rescoring large models enable
with `rescore/` and `rnnlm/`, and endpointing.

Note that vosk's result depends on how the audio is split into
`AcceptWaveform` calls (each call is processed in 0.2 s steps);
`accept_waveform` does the same, so feed the same pieces to compare.

#### (d) Streaming ASR

```bash
python3 decode_torch_streaming.py --model-dir ./torch-model-cn --wav ./zh.wav
```

The stream is fed 0.2 s of audio per `accept_waveform` call
(`--feed-seconds`); the AM runs in the model's chunks (24 frames, 240
ms).  The frame counter of the partials is the input frames decoded so
far.  Expected ending:

```
=== run 3/3 ===
  partial [24 frames]: ''
  partial [48 frames]: '不要 问'
  partial [72 frames]: '不要 问 你'
  partial [96 frames]: '不要 问 你 的'
  partial [120 frames]: '不要 问 你 的 国家'
  partial [144 frames]: '不要 问 你 的 国家 能'
  partial [168 frames]: '不要 问 你 的 国家 能 为你 做些什么'
  partial [192 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [216 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [240 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [264 frames]: '不要 问 你 的 国家 能 为你 做 什么 而'
  partial [288 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问'
  partial [312 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你你'
  partial [336 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能'
  partial [360 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你'
  partial [384 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的'
  partial [408 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家'
  partial [432 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使'
  partial [456 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使'
run 3: 0.15 s  RTF=0.033  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
audio: 4.76 s, chunk: 24 frames (240 ms, one i-vector per chunk), L/R context: 26/14 (260/140 ms)
```

The tail `促使 呢` (the transcript says `做 什么`) is what vosk decodes
too: this small model is wrong there, and the recognizer reproduces it.
Partials may revise earlier output as more audio arrives (normal ASR
behavior); only the final line is stable.  `--no-partial` skips them
(each `get_result` call recomputes the best path).

All decode scripts run the wav **3 times by default** (`--repeat N`); run
1 pays the one-time lazy-graph warmup (the first decode in a process
expands the lazy `HCLr ∘ Gr` graph state-by-state).

#### (e) Non-streaming ASR

```bash
python3 decode_torch_non_streaming.py --model-dir ./torch-model-cn --wav ./zh.wav
```

`OfflineRecognizer` runs the same pipeline on the whole utterance (vosk
has no separate offline mode: one `AcceptWaveform` call with the whole
file is this pipeline in 0.2 s steps), so streaming and non-streaming
give the same text.  Expected ending:

```
run 1: 1.14 s  RTF=0.240  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
run 2: 0.15 s  RTF=0.032  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
run 3: 0.15 s  RTF=0.032  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
audio: 4.76 s, chunk: 24 frames (one i-vector per chunk), L/R context: 26/14 (260/140 ms)
```

> The same two commands with `decode_flat_streaming.py` /
> `decode_flat_non_streaming.py` run the **flattened** forward()
> (`am/am_flat.py`) instead of the node-graph interpreter — identical
> results (they are parity-checked at export time).

### 3. Export to ONNX

```bash
python3 export_onnx.py --in-dir ./torch-model-cn \
    --out-dir ./onnx-model-cn
```

Output directory structure (same layout again; `am/` now holds the ONNX
artifact):

```
./onnx-model-cn/
├── am/
│   ├── final.onnx            # float32 AM (opset 17, dynamic batch+time;
│   │                         #   single file, metadata embedded)
│   ├── final.int8.onnx       # dynamically quantized AM (QInt8 weights)
│   ├── model_meta.json       # dims, frame_subsampling_factor,
│   │                         #   left/right context, decimation fields
│   └── transition_model.mdl  # HMM side of final.mdl (decoder needs it)
├── conf/                     # copied unchanged
├── ivector/                  # copied unchanged
├── graph/                    # copied unchanged
├── log.md                    # export record (markdown): gates +
│                             #   decimation analysis + per-layer evidence
└── README                    # copied unchanged
```

The export computes the subsampling on the **decoder's row lattice**
(see the decimation note below), gates it against the full-rate
computation, runs a shape sweep + numeric check (onnxruntime vs the torch
model, several shapes), and writes the int8 model.  Everything is recorded
in `log.md` (markdown); the stdout summary:

```
date: 2026-11-25 15:56
source: vosk-model-small-cn-0.22  <-  torch-model-cn/am/am.pt
toolchain: torch 2.15.0, opset 17
model: input_dim=40 ivector_dim=40 num_pdfs=2112
subsampling: frame_subsampling_factor=3 left/right=26/14

wrote final.onnx: opset 17, dynamic batch+time

decimation analysis (output rows 0::3)

zone         nodes    MAC/row  rows kept   where
dense           25    595,520        3/3   idct .. tdnnf3.noop
transition       2    245,760    1/3,2/3   tdnnf4.linear, tdnnf4.affine
decimated       60  2,371,584        1/3   tdnnf4.relu .. output
cut: tdnnf3.noop + tdnnf4.affine -> decimated grid [rows 0::3]
compute: 3.21M x T -> 1.63M x T MAC   (49% saved, 2.0x AM)
note: transition rows not exploited in this export (exact row masks would save +4%)

decimation gate (decimated == full-rate[::3])

T= 100: max_abs=0.00e+00   bitwise equal
T=1000: max_abs=0.00e+00   bitwise equal
T=  33: max_abs=0.00e+00   bitwise equal
timing: full-rate 32.6 ms vs decimated 19.1 ms (T=1000, CPU, x1.7)

embedded metadata: source_model, sample_rate, input_dim, ivector_dim, num_pdfs, frame_subsampling_factor, acoustic_scale, left_context, right_context, decimated_zone, compute_ratio, recurrent, state_rows, state_dim, modulus

checks (onnxruntime vs torch)

B=1 T=100: out (1, 34, 2112)  onnx-vs-torch max_abs=1.05e-05  OK
B=2 T=1000: out (2, 334, 2112)  onnx-vs-torch max_abs=1.35e-05  OK
B=1 T=33: out (1, 11, 2112)  onnx-vs-torch max_abs=9.06e-06  OK

int8 (quantize_dynamic, QInt8 weights)

wrote final.int8.onnx (QInt8)
int8 check (onnxruntime vs torch, informational):
B=1 T=100: out (1, 34, 2112)  onnx-vs-torch max_abs=6.43e-01  OK
B=2 T=1000: out (2, 334, 2112)  onnx-vs-torch max_abs=1.06e+00  info
B=1 T=33: out (1, 11, 2112)  onnx-vs-torch max_abs=6.21e-01  OK

files

am/final.onnx               13.00 MB
am/final.int8.onnx           3.38 MB
am/model_meta.json           0.00 MB
am/transition_model.mdl      0.17 MB
log.md                     (this file)

done: onnx-model-cn
```

**Decimation (kaldi-style row sparsity).**  The decoder consumes only the
network rows `k*factor` (kaldi's row grid starts at input frame 0 --
`nnet-compile-looped.cc` steps t += factor, no offset); kaldi's computation-graph builder computes only
the rows those outputs depend on, per node.  The export reproduces that:
nodes whose time offsets are multiples of `factor` (here `[-3,0]`/`[0,3]`
from `tdnnf5` up, plus the per-row head) are evaluated directly on the
row lattice, everything else is computed full-rate and sliced at the
boundary (`tdnnf3.noop + tdnnf4.affine` here).  Values are **bitwise
equal** to computing everything and slicing (the decimation gate), the
AM computes 49% of the MACs (measured ~2x faster at T=1000; int8
compounds on top).  `factor` is read from the model's own
metadata -- never assumed.  Each model's zone split lands in `log.md`
(per-layer table with offsets and needed-rows evidence) and compactly in
`model_meta.json` as `decimated_zone` / `compute_ratio`.

Recurrent nets (LSTM, `recurrent=1` in the metadata; gigaspeech) are not
decimated: they are exported from the flat module, whose recurrent groups
are TorchScript loop modules, so the graph holds an ONNX `Loop` with a
dynamic trip count.  Their onnx-vs-torch check is strict at T=100/33;
at T=1000 on random features it only checks the shape, because the LSTM
amplifies kernel float noise there (see `nnet3/notes.md` §7b).  Their
graph also carries the LSTM **streaming state** (below); the export gates
a state handover (onnx vs torch) and chained windows vs one-shot.

ONNX contract (see `model_meta.json` for the exact dims):

* input `features`: float32 `[B, T, feat_dim]` — MFCC from `conf/mfcc.conf`,
  audio at **16 kHz, int16 scale**
* input `ivector`: float32 `[B, T, ivector_dim]` — one i-vector per
  input frame, aligned with `features`.  kaldi's online decoders (and
  vosk) compute the network in chunks of `frames_per_chunk` frames (24)
  and give each chunk its own i-vector: they rewrite the network's
  `ReplaceIndex(ivector, t, 0)` to `Round(ivector, 24)`.  Here that
  descriptor reads row `t` of the input, and the caller repeats each
  chunk's i-vector over its frames (`recognizer.IvectorChunker`).
* output `log_probs`: float32 `[B, ceil(T/factor), num_pdfs]` —
  the frame subsampling is computed **inside** the graph on the decoder's
  row lattice, so the output is one row per `factor` input frames (30 ms
  per row here).  `B` and
  `T` are dynamic.
* recurrent models only (`recurrent=1`): input `state` float32
  `[B, state_rows, state_dim]` and output `state_out` (same shape) --
  the LSTM state carried from one decode window to the next.  Feed zeros
  for the first window of an utterance (and for a one-shot run), then
  each window's `state_out` to the next.  The window must be laid out
  like the recognizer's: `round_up(left_context, factor)` frames of left
  context + `n` new frames + `right_context`; `state_out` is the state
  for the window starting `n` frames later (`n` is derived from `T`).
  Chained windows reproduce the one-shot output exactly
  (`nnet3/notes.md` §7b).

Both `.onnx` files carry the `model_meta.json` key-values as **embedded
metadata** (`metadata_props`, same convention as
`scripts/whisper/export_onnx.py`), so the C++ runtime needs no sidecar:
`sample_rate`, `left_context`, `right_context`,
`frame_subsampling_factor`, `acoustic_scale`, `input_dim`,
`ivector_dim`, `num_pdfs`, `decimated_zone`, `compute_ratio`,
`recurrent`, `state_rows`, `state_dim`, `modulus` (kaldi's
`Nnet::Modulus()`: the chunk is rounded up to a multiple of it), ...  The
Python recognizers read them back via
`session.get_modelmeta().custom_metadata_map`.  `final.int8.onnx` is
`quantize_dynamic` output (QInt8 weights); on the test models the
recognition result is usually identical to float32 (see 4(d)) —
`decode_onnx_*.py --int8` selects it.

### 4. Use the ONNX model

#### (a) Verify the ONNX model against the PyTorch model

```bash
python3 - <<'EOF'
import numpy as np
import torch
import onnxruntime as ort
from recognizer import TorchAm, round_up

torch_am = TorchAm("./torch-model-cn")
spec = torch_am.net.spec
rng = np.random.default_rng(0)

# a window of 100 frames plus the receptive-field context (the net is
# pure: the caller supplies the context), one i-vector per frame
pad_left = round_up(spec.left_context, 3)
T = pad_left + 100 + spec.right_context
feats = rng.standard_normal((T, spec.input_dim)).astype(np.float32)
ivector = (rng.standard_normal((T, spec.ivector_dim)) * 0.1).astype(np.float32)

lp_torch, _ = torch_am.run(feats, ivector)  # rows ::3

sess = ort.InferenceSession("./onnx-model-cn/am/final.onnx",
                            providers=["CPUExecutionProvider"])
lp_onnx = sess.run(["log_probs"],
                   {"features": feats[None], "ivector": ivector[None]})[0][0]

# compare the rows of the real frames (both are rows ::3 of the window)
a = lp_torch[pad_left // 3 : pad_left // 3 + 34]
b = lp_onnx[pad_left // 3 : pad_left // 3 + 34]
print("rows", len(a), "max|torch - onnx| =", float(np.abs(a - b).max()))
EOF
```

Expected ending (float noise):

```
rows 34 max|torch - onnx| = 9.179115295410156e-06
```

For a recurrent model (`recurrent=1` in `model_meta.json`, e.g.
gigaspeech) the graph also takes the LSTM state: add
`"state": np.zeros((1, meta["state_rows"], meta["state_dim"]), np.float32)`
to the feed (zeros = the start of an utterance).  For streaming,
feed each window's `state_out` to the next window; `recognizer.py` does
this for you (`stream.am_state`).

#### (b) Streaming ASR

```bash
python3 decode_onnx_streaming.py --model-dir ./onnx-model-cn --wav ./zh.wav
```

Expected ending (partials as in Example 1(d)):

```
  partial [432 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使'
  partial [456 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使'
run 3: 0.06 s  RTF=0.013  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
audio: 4.76 s, chunk: 24 frames (240 ms, one i-vector per chunk), L/R context: 26/14 (260/140 ms)
```

With `--no-partial`:

```
run 3: 0.06 s  RTF=0.013  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
audio: 4.76 s, chunk: 24 frames (240 ms, one i-vector per chunk), L/R context: 26/14 (260/140 ms)
```

#### (c) Non-streaming ASR

```bash
python3 decode_onnx_non_streaming.py --model-dir ./onnx-model-cn --wav ./zh.wav
```

Expected ending:

```
run 1: 1.06 s  RTF=0.223  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
run 2: 0.06 s  RTF=0.013  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
run 3: 0.06 s  RTF=0.013  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
audio: 4.76 s, chunk: 24 frames (one i-vector per chunk), L/R context: 26/14 (260/140 ms)
```

#### (d) The int8 model

```bash
python3 decode_onnx_non_streaming.py --model-dir ./onnx-model-cn \
    --wav ./zh.wav --int8
python3 decode_onnx_streaming.py --model-dir ./onnx-model-cn \
    --wav ./zh.wav --int8
```

On this model the quantized AM decodes the same text as float32:

```
# non-streaming, int8 (tail of the run)
run 3: 0.07 s  RTF=0.015  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
audio: 4.76 s, chunk: 24 frames (one i-vector per chunk), L/R context: 26/14 (260/140 ms)

# streaming, int8 (tail of the run)
run 3: 0.07 s  RTF=0.014  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 促使 呢
audio: 4.76 s, chunk: 24 frames (240 ms, one i-vector per chunk), L/R context: 26/14 (260/140 ms)
```

That is not guaranteed: quantization can flip words of a weak model (it
does for small-ko, small-uk-v3-nano, small-vn), which is why the upload
pipeline (`upload-onnx/test.py`) checks int8 against vosk too and ships
it only when it matches.

### File sizes (vosk-model-small-cn-0.22)

| file | size | note |
|---|---|---|
| `vosk-model-small-cn-0.22/am/final.mdl` | 15.17 MB | original kaldi AM (export input) |
| `onnx-model-cn/am/transition_model.mdl` | 0.16 MB | HMM side, kept for the decoder |
| `onnx-model-cn/am/final.onnx` | 13.00 MB | float32 AM (single file, metadata inside) |
| `onnx-model-cn/am/final.int8.onnx` | 3.38 MB | QInt8 — 3.8x smaller |

### 5. Example 2 — English (vosk-model-small-en-us-0.15)

The exact same flow, with the English model and `en.wav`:

```bash
# (a) download
curl -L -o model-en.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -q model-en.zip   # -> ./vosk-model-small-en-us-0.15/
curl -L -o en.wav https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav

# (b) export to PyTorch
python3 export_torch.py --in-dir ./vosk-model-small-en-us-0.15 --out-dir ./torch-model-en

# (c) verify against kaldi
python3 compare_torch.py --model-dir ./vosk-model-small-en-us-0.15 --wav ./en.wav \
    --am-pt ./torch-model-en/am/am.pt

# (d) streaming ASR
python3 decode_torch_streaming.py --model-dir ./torch-model-en --wav ./en.wav

# (e) non-streaming ASR
python3 decode_torch_non_streaming.py --model-dir ./torch-model-en --wav ./en.wav

# 3. export to ONNX
python3 export_onnx.py --in-dir ./torch-model-en --out-dir ./onnx-model-en

# 4(b)(c) streaming / non-streaming ASR with the ONNX model
python3 decode_onnx_streaming.py --model-dir ./onnx-model-en --wav ./en.wav
python3 decode_onnx_non_streaming.py --model-dir ./onnx-model-en \
    --wav ./en.wav
```

The output directory structure is identical to Example 1.  All six
decode paths (torch/onnx x non-streaming/streaming, float32 and int8) end
with the same text, which is also the vosk package's.  Last run of each:

```
$ python3 decode_torch_streaming.py ... --no-partial
run 3: 0.10 s  RTF=0.027  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_torch_non_streaming.py ...
run 3: 0.10 s  RTF=0.027  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_streaming.py ... --no-partial
run 3: 0.03 s  RTF=0.008  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_non_streaming.py ...
run 3: 0.03 s  RTF=0.008  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_non_streaming.py ... --int8
run 3: 0.04 s  RTF=0.009  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_streaming.py ... --int8 --no-partial
run 3: 0.04 s  RTF=0.009  -> ask not what your country can do for you ask what you can do for your country
```

### File sizes (vosk-model-small-en-us-0.15)

| file | size | note |
|---|---|---|
| `vosk-model-small-en-us-0.15/am/final.mdl` | 15.22 MB | original kaldi AM (export input) |
| `onnx-model-en/am/transition_model.mdl` | 0.14 MB | HMM side, kept for the decoder |
| `onnx-model-en/am/final.onnx` | 13.03 MB | float32 AM (single file, metadata inside) |
| `onnx-model-en/am/final.int8.onnx` | 3.39 MB | QInt8 — 3.8x smaller |

## Other models / batch export

`run.sh` is a thin wrapper that exports several model dirs at once (it
does **not** download anything):

```sh
./run.sh ~/models/vosk-model-small-cn-0.22 ~/models/vosk-model-small-de-0.15 \
         --out-dir ./export
```

Validated end-to-end (converter outputs match kaldi to ~1e-5, cosine 1.0,
decoded texts correct):

| model | notes |
|---|---|
| `vosk-model-small-cn-0.22` | 41 MB zip |
| `vosk-model-small-en-us-0.15` | 39 MB |
| `vosk-model-small-de-0.15` | 40 MB (30→40 dim LDA) |
| `vosk-model-cn-0.22` | 1.3 GB (large graph `HCLG.fst`) |
| `vosk-model-en-us-0.22` | 1.8 GB (also `...-lgraph` works — same AM, small graph) |
| `vosk-model-en-us-0.42-gigaspeech` | 3.8 GB unzipped; recurrent (CNN + attention + 2 clockwork LSTMs); recurrent-net gate (median 1.4e-5, argmax 1.000) — see `nnet3/notes.md` §7b |

## Model directory layouts

vosk-api accepts two model directory layouts (see `Model::ConfigureV1` /
`Model::ConfigureV2` in `vosk-api/src/model.cc`):

* **standard** (v2) — `am/final.mdl`, `conf/model.conf`, `conf/mfcc.conf`,
  `graph/` (`HCLr.fst` + `Gr.fst`, or `HCLG.fst`), `ivector/`.  Most
  models ship this way (e.g. `vosk-model-small-cn-0.22`).
* **flat** (v1) — `final.mdl`, `mfcc.conf`, `HCLr.fst`, `Gr.fst`,
  `disambig_tid.int`, `word_boundary.int` at the TOP level (plus
  `ivector/`), and **no** `conf/` and **no** `model.conf` at all; the
  vosk runtime hardcodes its decoding defaults for these
  (`Model::ConfigureV1`).

The tools here expect the standard layout, so a flat model dir must be
rearranged first — `normalize_model_dir.py` does that in place:

```bash
python3 normalize_model_dir.py --model-dir ./vosk-model-small-de-zamia-0.3
```

It moves the files into `am/ conf/ graph/` (the exact v1→v2 mapping of
`ConfigureV1`/`ConfigureV2`) and, when the model has no `model.conf`,
writes one with EXACTLY the vosk v1 defaults (`--max-active=7000
--beam=13.0 --lattice-beam=6.0 --acoustic-scale=1.0
--frame-subsampling-factor=3`) so the recognition is unchanged by the
rearrange.  It is idempotent and leaves standard model dirs alone; run
it before `export_torch.py`.  Among the models exported by
`upload-onnx/` exactly two are flat: `vosk-model-small-de-zamia-0.3` and
`vosk-model-small-fr-pguyot-0.3` (that pipeline runs the normalizer
automatically after download).

## Readable model code (am_flat.py)

`export_torch.py` writes `am/am_flat.py` next to `am/am.pt`: an
auto-generated, self-contained `nn.Module` ("FlatAm") that spells the
network out as plain python — one named attribute per kaldi component and
a `forward()` with the node-graph connections written literally, in the
order kaldi wrote them.  E.g. the i-vector join reads
`self.input2(torch.cat([x_delta, x_ivector], dim=-1))` (the ivector input is per frame, `[B, T, S]`)
and a TDNNF residual reads
`self.tdnnf2_noop((0.75 * x_tdnn1_dropout) + x_tdnnf2_dropout)`.

Use it to *read* the model (what connects to what), or to *edit* it
(prune layers, insert hooks, experiment) — the wiring is code, not data.
The per-component math modules come from `torch_model` (`build_component`),
so there is one implementation to maintain.  Semantics are identical to
`Nnet3Torch` (the graph interpreter the other tools use) — the generated
module is parity-checked at several shapes before it is written
(measured 0.0).

```python
import sys; sys.path.insert(0, "path/to/sherpa-onnx/scripts/vosk")
from am_flat import FlatAm

net = FlatAm.from_am_pt("./torch-model-cn/am/am.pt")
log_probs = net(features, ivector)   # [B, T, num_pdfs], full 10 ms grid
```

Notes:

* output is **full-rate** (one row per 10 ms input frame); the 3x
  subsampling for decoding is applied outside.
* the receptive-field edge padding (replicated edge frames) is visible at
  the top of `forward()` and the matching crop at the bottom — see
  "Notes on semantics".
* complex descriptor lines keep the original kaldi expression as a
  trailing comment (`# Append(delta, ReplaceIndex(ivector, t, 0))`).
* recurrent groups (LSTM cycles) are separate classes `_Rec0`, `_Rec1`,
  ... at the end of the file, compiled with `torch.jit.script` (so the
  ONNX export keeps their per-row loop as a `Loop`); they own their
  components, so those weights sit under `_rec<i>.` in the state_dict.
  Such a net also has `initial_state(batch)` and
  `forward_streaming(features, ivector, state, shift)` (the streaming
  state, see `nnet3/notes.md` §7b); `forward()` is the one-shot run with
  the blank state.

Regenerate or re-check any time:

```sh
python3 flatten.py --am-pt ./torch-model-cn/am/am.pt
```

## Verifying against kaldi

```sh
# 1) converter gate: the torch model must equal the original kaldi model
python3 compare_torch.py --model-dir vosk-model-small-cn-0.22 --wav zh.wav
python3 compare_torch.py --model-dir <model> --wav x.wav --am-pt <export>/am/am.pt
#    (the second form checks the reload path: state_dict + args)

# 2) the exported ONNX: AM parity vs kaldi + wav -> text through the
#    full python stack (features -> i-vector -> ONNX -> decode -> text)
python3 compare_onnx.py --model-dir export/vosk-model-small-cn-0.22 \
    --ref-model-dir vosk-model-small-cn-0.22 --wav zh.wav \
    --expect '不要问你的国家能为你做什么而要问你能为你的国家做什么'
```

Both run kaldi's `nnet3-compute` with `--frames-per-chunk=1000000` (the
whole utterance as one chunk): its default 50-frame chunks restart a
recurrent net's LSTM state every chunk.  For recurrent nets the gate is
median per-row max <= 1e-3 plus argmax agreement >= 0.99 instead of the
worst max, which float noise amplified by the LSTM can exceed even for
kaldi against itself (`nnet3/notes.md` §7b).

`compare_onnx.py` runs two stages: stage 1 compares `final.onnx` row-by-row
against kaldi's `nnet3-compute` on identical inputs; stage 2 decodes a real
wav to text and gates on `--expect` (whitespace-insensitive).  It also
prints the `vosk` package's 1-best for the same model and wav (dither
off, no lattice rescoring) and whether it equals ours -- the comparison
`upload-onnx/test.py` gates on.  (The vosk wheel runs in a subprocess;
its bundled kaldi clashes in-process.)  Test wavs:
[zh.wav](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/zh.wav),
[en.wav](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav),
[de.wav](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/de.wav).

## Notes on semantics

* audio is read and resampled **in memory** to the model's sample rate
  (`sample_rate` in `model_meta.json`) with **librosa** (deterministic),
  so every consumer sees the same samples; any input rate is fine.  (The kaldi-CLI verification writes a temp 16 kHz wav -- those
  tools only take file paths.)  (Not sox:
  its 16-bit output is dithered, which perturbs the i-vector enough to
  flip marginal decode tails.)
* features are computed at **int16 scale** (vosk's convention), dither 0.
* small models decode over a lazy `HCLr ∘ Gr` composition
  (`kaldifst.lookahead_compose_fst`, like vosk-api); large models load
  `graph/HCLG.fst` directly.  Never materialize HCLr∘Gr into a VectorFst —
  it is impractically slow.
* `left_context`/`right_context` in `model_meta.json` are the model's
  receptive field in 10 ms frames (e.g. 26/14 for cn/en).  The exported
  net is pure: the caller supplies that context (edge frames are
  replicated -- kaldi's edge rule).  The recognizer computes the AM in
  kaldi's looped-decodable chunks (`frames_per_chunk`, 24 frames): chunk
  `c` runs on the window `[24c - round_up(L, 3), 24(c+1) + R)` and keeps
  its own 8 rows; output row k corresponds to input frame `3k` (30 ms per
  row).  The right context is inherent latency (140 ms for cn/en).
* the i-vector changes per chunk: chunk `c` takes the latest estimate
  when it is computed, and input frame `t` uses the i-vector of the chunk
  that first needed frame `24 * floor(t / 24)` -- what kaldi's rewrite of
  `ReplaceIndex(ivector, t, 0)` to `Round(ivector, 24)` means for a
  looped computation (`recognizer.IvectorChunker`).
* the decoder reads the model's own `conf/model.conf` (`--beam`,
  `--max-active`, `--min-active`, `--lattice-beam`, `--frames-per-chunk`,
  `--acoustic-scale`, `--endpoint.silence-phones`); unset keys keep
  kaldi's defaults -- the same as the vosk runtime.
* recurrent (LSTM) models: each stream carries the AM's LSTM state from
  one chunk to the next (`stream.am_state`, like kaldi's looped
  decodable; `nnet3/notes.md` §7b).
* decoding needs the i-vector extractor (`ivector/`) and the
  `am/transition_model.mdl` sidecar at runtime.
* RNNLM rescoring (`rnnlm/` in large models) is not covered here.

Design details of the nnet3 reader: see `nnet3/notes.md`.
