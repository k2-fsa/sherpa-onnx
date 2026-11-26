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
             kaldi-native-fbank kaldi-hmm-gmm kaldi-native-io kaldifst onnxscript \
             'kaldi-decoder>=0.6.0'
```

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

`kaldi-decoder >= 0.6.0` is required: older wheels export their bundled
openfst symbols, whose FST-type registry then breaks reading the vosk
graph types (`olabel_lookahead`/`ngram` in HCLr.fst/Gr.fst) when loaded
alongside kaldifst.  0.6.0 hides them.  `kaldi-hmm-gmm` (1.3.5) still
exports its copy, so `kaldifst` must be imported **before**
`kaldi_hmm_gmm` (recognizer.py does; otherwise reading HCLr.fst fails with
`Unknown FST type olabel_lookahead`).

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

All six decode scripts are built on `recognizer.py`, a python port of
the sherpa-onnx recognizer design (`csrc/online-recognizer.h`,
`offline-recognizer.h`, `online-stream.h`): `OnlineStream`/`OfflineStream`
accept samples and compute features + i-vector, `Decoder` is a kaldi FST
lattice decoder fed one model-output row at a time, and
`OnlineRecognizer`/`OfflineRecognizer` test readiness and decode once:

```python
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
`am/am_flat.py`); streams, windowing and decoding are shared.

`OnlineRecognizer` also implements the sherpa-onnx fixed-chunk scheme
(`--chunk-size N`, the recognizer's `chunk_size`): one
decode step runs the model on `L + N + R` frames and keeps the middle
`N` frames' rows, ready as soon as
`num_frames_ready > num_processed_frames + N + R` (the left `L` frames
come from the stream's history or, at the very start, kaldi-style
replicated-edge padding).  `N` must be a multiple of
`frame_subsampling_factor` (rows anchor every factor x 10 ms).

#### (d) Streaming ASR

```bash
python3 decode_torch_streaming.py --model-dir ./torch-model-cn \
    --wav ./zh.wav --chunk-size 30
```

The stream is fed 0.2 s of audio at a time (a pacing choice — any
value works, see the driver) and decoded in `--chunk-size` steps: one
step runs the model on `L + chunk_size + R` frames and the rows go
straight into the lattice — the ring-buffer convention kaldi's looped
decodable uses (see "Notes on semantics").  The frame counter in the
partials is the input frames decoded so far.  Expected output
(intermediate logs included):

```
WARNING: ignoring mfcc.conf key allow-upsample=true
WARNING: ignoring mfcc.conf key allow-downsample=true
=== run 1/3 ===
  partial [30 frames]: '不要'
  partial [60 frames]: '不要 问 你'
  partial [90 frames]: '不要 问 你 的 家'
  partial [120 frames]: '不要 问 你 的 国家'
  partial [150 frames]: '不要 问 你 的 国家 能 为'
  partial [180 frames]: '不要 问 你 的 国家 能 为你 做'
  partial [210 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [240 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [270 frames]: '不要 问 你 的 国家 能 为你 做 什么 而已'
  partial [300 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你'
  partial [330 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能为'
  partial [360 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的'
  partial [390 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家'
  partial [420 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 措施'
  partial [450 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
  partial [474 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
run 1: 1.17 s  RTF=0.247  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
=== run 2/3 ===
  partial [30 frames]: '不要'
  partial [60 frames]: '不要 问 你'
  partial [90 frames]: '不要 问 你 的 家'
  partial [120 frames]: '不要 问 你 的 国家'
  partial [150 frames]: '不要 问 你 的 国家 能 为'
  partial [180 frames]: '不要 问 你 的 国家 能 为你 做'
  partial [210 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [240 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [270 frames]: '不要 问 你 的 国家 能 为你 做 什么 而已'
  partial [300 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你'
  partial [330 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能为'
  partial [360 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的'
  partial [390 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家'
  partial [420 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 措施'
  partial [450 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
  partial [474 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
run 2: 0.30 s  RTF=0.063  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
=== run 3/3 ===
  partial [30 frames]: '不要'
  partial [60 frames]: '不要 问 你'
  partial [90 frames]: '不要 问 你 的 家'
  partial [120 frames]: '不要 问 你 的 国家'
  partial [150 frames]: '不要 问 你 的 国家 能 为'
  partial [180 frames]: '不要 问 你 的 国家 能 为你 做'
  partial [210 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [240 frames]: '不要 问 你 的 国家 能 为你 做 什么'
  partial [270 frames]: '不要 问 你 的 国家 能 为你 做 什么 而已'
  partial [300 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你'
  partial [330 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能为'
  partial [360 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的'
  partial [390 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家'
  partial [420 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 措施'
  partial [450 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
  partial [474 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
run 3: 0.31 s  RTF=0.064  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
audio: 4.76 s, chunk: 30 frames (300 ms), L/R context: 26/14 (260/140 ms)
```

Notes on the intermediate logs:

* **Partials may revise earlier output** — e.g. `国际` later becomes
  `国家`, `促使` becomes `做 什么`.  This is normal ASR behavior: each
  partial is the 1-best hypothesis over the evidence so far, and more
  audio + the language model can re-rank what came before.  Only the
  final line after the last chunk is stable.
* decoding is incremental: each chunk advances a persistent
  `LatticeFasterDecoder` once (no re-decoding of the past).
* `--no-partial` suppresses the partial lines.

Streaming uses the **online (running) i-vector**, like a real streamer.
Each decode step uses the estimate at the last frame of its window (as
kaldi's looped decodable does): the estimator is fed only up to there, so
the result does not depend on how the audio is split into
`accept_waveform()` calls (verified bitwise, 0.01 s .. whole-file feeds).
The `--max-count` flag (default 100, same as the vosk runtime; 0
disables) tunes how strongly the i-vector prior is scaled with
accumulating data.  The decode tail is robust to it: verified with
`--max-count 0/100/1000` x chunk sizes 15/30/60 on both models, with no
end-of-utterance padding.

All decode scripts run the wav **3 times by default** (`--repeat N`).
Every run prints its full intermediate logs (partials etc., as in the
expected output above) and a summary line with time and text.  Run 1
pays the one-time lazy-graph warmup; runs 2-3 show the steady state.

The stats line shows **audio duration, elapsed decode time and RTF**
(elapsed / audio).  Setup is excluded from `elapsed`: model and graph
loading, library imports, audio file I/O, and the per-utterance frontend
(feature extraction + i-vector).  Note the *first* decode
in a process expands the lazy `HCLr ∘ Gr` graph state-by-state (~2 s for
this model) — also initialization, paid once per process.

Where the time goes (small-cn, chunk 30, steady state): extracting the
per-chunk partial hypothesis (`get_best_path`) and the online i-vector
dominate (~40% each), lattice advance ~13%, and the AM itself only ~5% —
this python pipeline is bound by bookkeeping, not the neural net.
`--no-partial` skips partial extraction entirely (0.16 s, RTF 0.035).
The AM's share grows with model size and in a C++ deployment; that is
where the exported AM's decimation (section 3) pays off.

For reference, on the same machine and wav (4.76 s, small-cn), the
`vosk` package decodes in 0.66 s (RTF 0.14) while this pipeline's steady
state is 0.23 s (RTF 0.048) with per-chunk partials -- 0.16 s (RTF 0.035)
without.  vosk's streaming tail degrades (`促使 呢`)
under its own i-vector settings; ours decodes the tail correctly.

#### (e) Non-streaming ASR

```bash
python3 decode_torch_non_streaming.py --model-dir ./torch-model-cn --wav ./zh.wav
```

The whole utterance goes through the AM in one call.  Expected ending:

```
WARNING: ignoring mfcc.conf key allow-upsample=true
WARNING: ignoring mfcc.conf key allow-downsample=true
run 1: 0.91 s  RTF=0.190  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
run 2: 0.05 s  RTF=0.010  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
run 3: 0.04 s  RTF=0.009  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
audio: 4.76 s, chunk: one-shot (whole utterance), L/R context: 26/14 (260/140 ms)
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

embedded metadata: source_model, sample_rate, input_dim, ivector_dim, num_pdfs, frame_subsampling_factor, acoustic_scale, left_context, right_context, decimated_zone, compute_ratio

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

ONNX contract (see `model_meta.json` for the exact dims):

* input `features`: float32 `[B, T, feat_dim]` — MFCC from `conf/mfcc.conf`,
  audio at **16 kHz, int16 scale**
* input `ivector`: float32 `[B, ivector_dim]` — one i-vector per utterance
  (broadcast over time)
* output `log_probs`: float32 `[B, ceil(T/factor), num_pdfs]` —
  the frame subsampling is computed **inside** the graph on the decoder's
  row lattice, so the output is one row per `factor` input frames (30 ms
  per row here).  `B` and
  `T` are dynamic.

Both `.onnx` files carry the `model_meta.json` key-values as **embedded
metadata** (`metadata_props`, same convention as
`scripts/whisper/export_onnx.py`), so the C++ runtime needs no sidecar:
`sample_rate`, `left_context`, `right_context`,
`frame_subsampling_factor`, `acoustic_scale`, `input_dim`,
`ivector_dim`, `num_pdfs`, `decimated_zone`, `compute_ratio`, ...  The
Python recognizers read them back via
`session.get_modelmeta().custom_metadata_map`.  `final.int8.onnx` is
`quantize_dynamic` output (QInt8 weights); on the test models the
recognition result is identical to float32 — `decode_onnx_*.py --int8`
selects it.

### 4. Use the ONNX model

#### (a) Verify the ONNX model against the PyTorch model

```bash
python3 - <<'EOF'
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from recognizer import OfflineStream, load_audio, mfcc_options, read_meta, round_up
from torch_model import Nnet3Torch

model_dir = Path("./torch-model-cn")
meta = read_meta(model_dir)
samples, sr = load_audio("./zh.wav", meta["sample_rate"])
stream = OfflineStream(model_dir, mfcc_options(model_dir / "conf" / "mfcc.conf"))
stream.accept_waveform(sr, samples)
stream.input_finished()
feats = stream.get_frames(0, stream.num_frames_ready)
iv = stream.ivector

# the net is pure: feed the receptive-field context yourself (kaldi's
# replicated-edge rule) and keep the rows of the real frames
pad_left = round_up(meta["left_context"], 3)
padded = np.pad(feats, ((pad_left, meta["right_context"]), (0, 0)), mode="edge")

payload = torch.load(model_dir / "am" / "am.pt", map_location="cpu",
                     weights_only=False)
net = Nnet3Torch.from_export_args(payload["args"])
net.load_state_dict(payload["state_dict"])
net.eval()
with torch.no_grad():
    lp_torch = net(torch.from_numpy(padded)[None],
                   torch.from_numpy(iv)[None])[0].numpy()
lp_torch = lp_torch[pad_left : pad_left + len(feats) : 3]

sess = ort.InferenceSession("./onnx-model-cn/am/final.onnx",
                            providers=["CPUExecutionProvider"])
lp_onnx = sess.run(["log_probs"],
                   {"features": padded[None], "ivector": iv[None]})[0][0]
lp_onnx = lp_onnx[pad_left // 3 :]

n = min(len(lp_torch), len(lp_onnx))
print("rows", n, "max|torch - onnx| =",
      float(np.abs(lp_torch[:n] - lp_onnx[:n]).max()))
EOF```

Expected ending:

```
WARNING: ignoring mfcc.conf key allow-upsample=true
WARNING: ignoring mfcc.conf key allow-downsample=true
rows 158 max|torch - onnx| = 1.3709068298339844e-05
```

#### (b) Streaming ASR

```bash
python3 decode_onnx_streaming.py --model-dir ./onnx-model-cn \
    --wav ./zh.wav --chunk-size 30
```

Expected ending:

```
  partial [450 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
  partial [474 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
run 3: 0.23 s  RTF=0.048  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
audio: 4.76 s, chunk: 30 frames (300 ms), L/R context: 26/14 (260/140 ms)

```

(Partial hypotheses print as the stream progresses, as in Example 1(d).
The frame counter is the input frames decoded so far, e.g. `[30 frames]`
after the first decode step.)

`get_result` is **expensive** — each call re-runs the lattice's
`get_best_path`, and its cost grows with the lattice size (it is the
about 40% of the cn streaming time).
The per-chunk partials above call it after every chunk; for throughput,
call it at a **low cadence** (e.g. once per second) or skip it entirely
with `--no-partial`, and the RTF drops sharply:

```
$ python3 decode_onnx_streaming.py --model-dir ./onnx-model-cn --wav ./zh.wav --chunk-size 30 --no-partial
run 2: 0.16 s  RTF=0.034  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
run 3: 0.16 s  RTF=0.035  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
audio: 4.76 s, chunk: 30 frames (300 ms), L/R context: 26/14 (260/140 ms)

$ python3 decode_onnx_streaming.py --model-dir ./onnx-model-en --wav ./en.wav --chunk-size 30 --no-partial
run 2: 0.09 s  RTF=0.023  -> ask not what your country can do for you ask what you can do for your country
run 3: 0.09 s  RTF=0.023  -> ask not what your country can do for you ask what you can do for your country
audio: 3.85 s, chunk: 30 frames (300 ms), L/R context: 26/14 (260/140 ms)
```

(cn: RTF 0.048 -> 0.035; en: 0.036 -> 0.023.  Every call to
`get_result` costs the same whether or not you print it — do not call it
more often than the UI needs.)

#### (c) Non-streaming ASR

```bash
python3 decode_onnx_non_streaming.py --model-dir ./onnx-model-cn \
    --wav ./zh.wav
```

Expected ending:

```
WARNING: ignoring mfcc.conf key allow-upsample=true
WARNING: ignoring mfcc.conf key allow-downsample=true
run 1: 0.87 s  RTF=0.184  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
run 2: 0.04 s  RTF=0.008  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
run 3: 0.03 s  RTF=0.007  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
audio: 4.76 s, chunk: one-shot (whole utterance), L/R context: 26/14 (260/140 ms)
```


#### (d) The int8 model — identical recognition, ~1/4 the size

```bash
python3 decode_onnx_non_streaming.py --model-dir ./onnx-model-cn \
    --wav ./zh.wav --int8
python3 decode_onnx_streaming.py --model-dir ./onnx-model-cn \
    --wav ./zh.wav --chunk-size 30 --int8
```

Test result: the quantized model decodes **exactly the same text** as the
float32 model, non-streaming and streaming (the quantization noise, <= ~1 in
the logit domain, never flips a word here):

```
# non-streaming, int8 (tail of the run)
run 3: 0.03 s  RTF=0.007  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
audio: 4.76 s, chunk: one-shot (whole utterance), L/R context: 26/14 (260/140 ms)

# streaming, int8 (tail of the run)
  partial [474 frames]: '不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么'
run 3: 0.23 s  RTF=0.049  -> 不要 问 你 的 国家 能 为你 做 什么 而 要 问 你 能 为你 的 国家 做 什么
audio: 4.76 s, chunk: 30 frames (300 ms), L/R context: 26/14 (260/140 ms)
```

### File sizes (vosk-model-small-cn-0.22)

| file | size | note |
|---|---|---|
| `vosk-model-small-cn-0.22/am/final.mdl` | 15.17 MB | original kaldi AM (export input) |
| `onnx-model-cn/am/transition_model.mdl` | 0.16 MB | HMM side, kept for the decoder |
| `onnx-model-cn/am/final.onnx` | 13.00 MB | float32 AM (single file, metadata inside) |
| `onnx-model-cn/am/final.int8.onnx` | 3.38 MB | QInt8 — 3.8x smaller, same results |

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
python3 decode_torch_streaming.py --model-dir ./torch-model-en \
    --wav ./en.wav --chunk-size 30

# (e) non-streaming ASR
python3 decode_torch_non_streaming.py --model-dir ./torch-model-en --wav ./en.wav

# 3. export to ONNX
python3 export_onnx.py --in-dir ./torch-model-en --out-dir ./onnx-model-en

# 4(b)(c) streaming / non-streaming ASR with the ONNX model
python3 decode_onnx_streaming.py --model-dir ./onnx-model-en \
    --wav ./en.wav --chunk-size 30
python3 decode_onnx_non_streaming.py --model-dir ./onnx-model-en \
    --wav ./en.wav
```

The output directory structure is identical to Example 1.  All six
decode paths (torch/onnx x non-streaming/streaming, float32 and int8) end
with the same text.  Last run of each:

```
$ python3 decode_torch_streaming.py ... --chunk-size 30
run 3: 0.18 s  RTF=0.047  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_torch_non_streaming.py ...
run 3: 0.02 s  RTF=0.006  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_streaming.py ... --chunk-size 30
run 3: 0.14 s  RTF=0.036  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_non_streaming.py ...
run 3: 0.01 s  RTF=0.004  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_non_streaming.py ... --int8
run 3: 0.01 s  RTF=0.004  -> ask not what your country can do for you ask what you can do for your country

$ python3 decode_onnx_streaming.py ... --chunk-size 30 --int8
run 3: 0.15 s  RTF=0.038  -> ask not what your country can do for you ask what you can do for your country
```

### File sizes (vosk-model-small-en-us-0.15)

| file | size | note |
|---|---|---|
| `vosk-model-small-en-us-0.15/am/final.mdl` | 15.22 MB | original kaldi AM (export input) |
| `onnx-model-en/am/transition_model.mdl` | 0.14 MB | HMM side, kept for the decoder |
| `onnx-model-en/am/final.onnx` | 13.03 MB | float32 AM (single file, metadata inside) |
| `onnx-model-en/am/final.int8.onnx` | 3.39 MB | QInt8 — 3.8x smaller, same results |

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

## Readable model code (am_flat.py)

`export_torch.py` writes `am/am_flat.py` next to `am/am.pt`: an
auto-generated, self-contained `nn.Module` ("FlatAm") that spells the
network out as plain python — one named attribute per kaldi component and
a `forward()` with the node-graph connections written literally, in the
order kaldi wrote them.  E.g. the i-vector join reads
`self.input2(torch.cat([x_delta, _fix_index(x_ivector, 0)], dim=-1))`
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

`compare_onnx.py` runs two stages: stage 1 compares `final.onnx` row-by-row
against kaldi's `nnet3-compute` on identical inputs; stage 2 decodes a real
wav to text and gates on `--expect` (whitespace-insensitive).  It also
prints a `vosk` package baseline for reference — informational only (the
vosk wheel must run in a subprocess; its bundled kaldi clashes
in-process).  Test wavs:
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
  receptive field in 10 ms frames (e.g. 26/14 for cn/en).  When decoding a
  whole utterance (non-streaming) you can feed it in one go -- with the
  receptive-field context supplied by the caller (streams replicate edge
  frames; kaldi's edge rule).  When feeding **chunks** (streaming), you must also supply
  context — the ring-buffer approach kaldi's looped decodable uses; feeding
  a small chunk without context gives chunk-edge artifacts.  All streaming
  recognizers count input feature frames (row k at 10 ms frame `3k`) and
  feed the lattice directly — no repeat anywhere: to emit decoder rows
  `[d0, d1)` they run the AM on the 10 ms window `[3*d0 - left_context,
  3*(d1-1) + right_context]` (start snapped down to the `0::3` row grid) and keep the rows `[d0, d1)`.  What differs is only the AM
  output's row rate: `decode_onnx_streaming.py` gets one row per `factor` out of
  the graph (they go straight through), while the torch/flat AMs output
  full-rate (one row per 10 ms frame) and their `run()` keeps the `0::3`
  rows.
  The `right_context` also means streaming has that much inherent latency
  (e.g. 140 ms for cn/en).  `frame_subsampling_factor` says output
  row k corresponds to input frame `k*factor` (word times = row
  index x factor x 10 ms, 30 ms here);
  `acoustic_scale` feeds the decoder; `num_pdfs`/`ivector_dim` are sanity
  dims.
* the lattice decoder reads the model's own `conf/model.conf` decode
  params (`--beam` / `--max-active` / `--min-active` / `--lattice-beam`)
  -- the same overrides the vosk runtime applies at load time; keys the
  conf does not set keep the kaldi library defaults.
* chunk sizing for streaming: the only real knob is latency
  (`(N + R) x 10 ms` for the first frame of a chunk); overlap cost is
  `(L+R)/N`, so pick `N` as a multiple of 3 near the latency budget
  (30 frames = 300 ms is a good default).
* decoding needs the i-vector extractor (`ivector/`) and the
  `am/transition_model.mdl` sidecar at runtime.
* RNNLM rescoring (`rnnlm/` in large models) is not covered here.

Design details of the nnet3 reader: see `nnet3/notes.md`.
