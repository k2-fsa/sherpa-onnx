# Export X-ASR to AXERA AX650 (AXMODEL)

This directory contains the pipeline that converts the X-ASR encoder (a Zipformer2
transducer exported by icefall) into an `axmodel` that runs on the AXERA AX650 NPU.

The three sub-models keep their usual roles:

| Sub-model | Runs on | Note |
|---|---|---|
| `encoder` | **AX650 NPU** | this directory |
| `decoder`, `joiner` | CPU (onnxruntime) | tiny, and called once per frame |

## Why a plain ONNX export is not enough

`export-streaming-as-non-streaming-onnx.py` produces a graph that Pulsar2 cannot
compile directly. Three rewrites are required (all implemented in
`prepare-encoder.py`):

| Problem in the exported graph | Consequence in Pulsar2 | Fix |
|---|---|---|
| 43 `If` nodes (causal / non-causal branches in `ConvModule`, `downsample`) | quantization pass aborts with `Op Execution Error: .../If` | evaluate the conditions once (they only depend on the static input length) and inline the taken branch |
| 2477 dynamic shape tensors feeding `Reshape` (from `NonlinAttention`) | `Reshape, shapefn failed` | evaluate them and turn them into initializers |
| 99 `Log(1 + Exp(-abs(x)))` patterns (the `SwooshL/SwooshR` activations) | Pulsar2 has no `Log`; the graph is split into 17 NPU + 16 ONNX subgraphs, which is slow to compile and slow at runtime | `log(1 + exp(z)) == softplus(z)`, so replace them by `Softplus` (which Pulsar2 supports) → a single NPU subgraph |

After the rewrites the prepared graph is numerically identical to the exported
one (`cosine = 1.00000000` for the fixed input window).

## Step 1. Export the fp32 ONNX encoder

Inside icefall (`egs/librispeech/ASR`), with the X-ASR checkpoint
(`pretrained.pt`, Apache-2.0) placed in `--exp-dir` as `epoch-99.pt`:

```bash
./zipformer/export-streaming-as-non-streaming-onnx.py \
  --epoch 99 --avg 1 --use-averaged-model 0 \
  --exp-dir ./exp --tokens ./exp/tokens.txt \
  --max-len 1000 --dynamic-axes 0 --keep-x-lens 0 \
  --use-int32-inputs 0 --enable-int8-quantization 0 \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --causal 1 --chunk-size "-1" --left-context-frames "-1"
```

`--max-len 1000` fixes the input window to 10 s (1000 fbank frames at 16 kHz /
10 ms). Longer audio has to be segmented with VAD, like any other non-streaming
sherpa-onnx model. The exporter writes
`$exp_dir/encoder-epoch-99-avg-1.onnx`; copy it next to this README (or pass its
full path to the next step).

## Step 2. Prepare the graph for Pulsar2

```bash
# run this from the directory that contains this README
python3 prepare-encoder.py \
  --src /path/to/exp/encoder-epoch-99-avg-1.onnx \
  --dst encoder-ax650.onnx \
  --num-frames 1000
```

Expected output:

```
folded 43 If nodes
froze 2477 shape tensors
rewrote 99 Log nodes to Softplus
saved encoder-ax650.onnx (nodes: 4621)
output shape: [1, 248, 512]  cosine(prepared, original) = 1.00000000
```

## Step 3. Create the calibration dataset

```bash
pip install kaldi-native-fbank numpy

python3 make-calibration-set.py \
  --wav-dir /path/to/wavs \
  --output ./x4d.tar.gz \
  --num-frames 1000 --num-samples 8
```

Use real recordings (16 kHz, mono) that cover your acoustic conditions. The
script produces a `tar.gz` of float32 `.npy` files of shape `(1, 1, 1000, 80)`,
which is the format Pulsar2 expects for `calibration_format: Numpy`.

## Step 4. Compile with Pulsar2

```bash
pulsar2 build --config pulsar2-encoder.json
```

`pulsar2-encoder.json` uses INT8 (U8 activations, S8 weights) with MinMax
calibration. Keep `compiler.check` at `0`: with `1` the build spends a very long
time in the "confused op refine" pass on this graph.

The result is `output_encoder/encoder.axmodel` (about 165 MB for X-ASR).

## Step 5. Verify

`verify-with-simulator.py` takes one calibration sample, dumps it as the raw
float32 input that `pulsar2 run` expects, runs both models and reports the
cosine similarity:

```bash
python3 verify-with-simulator.py \
    --onnx encoder-ax650.onnx \
    --axmodel output_encoder/encoder.axmodel \
    --sample calib_x4d/0000.npy
```

```
cosine = 0.999073  MAE = 0.076495  shape = [1, 248, 512]
```

> `pulsar2 run` compares its output against any `<output>.bin` that already
> exists in `--output_dir` and asserts bit-exact equality, so always use a fresh
> directory (the script does that for you).

Reference numbers obtained with this pipeline (AX650 simulation, 4 test wavs):

| Metric | Value |
|---|---|
| cosine (mean / min) | 0.9981 / 0.9967 |
| MAE | 0.08 – 0.14 |
| greedy-search text | 1 of 4 utterances token-identical to fp32; the rest differ in a few English words |

## Deployment

The prebuilt models are published on HuggingFace
(<https://huggingface.co/yunyu1258/x-asr-axmodel>) together with a Python example
(`run_asr.py`, uses `axengine` + `numpy` only).

## Running this in CI / Docker

The whole pipeline also runs on a GitHub Actions runner:
`.github/workflows/export-x-asr-axera.yaml` (`workflow_dispatch`, ~1.5–2 h per run).
Two pieces of it are worth knowing about if you script it yourself.

### Which Pulsar2 version

This conversion was developed and verified with **Pulsar2 8.0** (the current release on
<https://huggingface.co/AXERA-TECH/Pulsar2>). Grab the *lite* package (830 MB, enough for
`pulsar2 build` / `pulsar2 run`):

```bash
curl -SL -O https://huggingface.co/AXERA-TECH/Pulsar2/resolve/main/8.0/ax_pulsar2_8.0_lite_package.tar.gz
tar xf ax_pulsar2_8.0_lite_package.tar.gz
export PATH=$PWD/ax_pulsar2_8.0_lite_package/bin:$PATH
```

Image `ghcr.io/csukuangfj/axera-pulsar2:4.2` is Pulsar2 4.2, i.e. several major versions
older than what was used here; if a container is preferred, build it from the 8.0 tarball
(the package is self-contained, no GPU and no extra apt packages are needed):

```dockerfile
FROM ubuntu:22.04
ADD ax_pulsar2_8.0_lite_package.tar.gz /opt/pulsar2
ENV PATH=/opt/pulsar2/ax_pulsar2_8.0_lite_package/bin:$PATH
```

### Two bugs in the 8.0 launcher scripts

The bundled wrappers have the wrong paths, so `pulsar2` fails with
`can't open file .../pulsar2/axnn/axnn/yamain/main.py`. Patch them after extraction:

```bash
P=ax_pulsar2_8.0_lite_package
# bin/pulsar2: axnn/axnn/yamain -> axnn/yamain
sed -i 's#/pulsar2/axnn/axnn/yamain#/pulsar2/axnn/yamain#' "$P/bin/pulsar2"
# bin/python3: the package ships ld-linux-x86-64.so.2, not ld-2.31.so
sed -i 's#ld-2.31.so#ld-linux-x86-64.so.2#' "$P/bin/python3"
pulsar2 version   # -> version: 8.0
```

(The license file is bundled and installed to `~/.hasplm` automatically by `bin/pulsar2`.)

### Measured result on GitHub Actions

Run: <https://github.com/yuyun2000/x-asr-axmodel/actions/runs/43004485940>

| Step | Time |
|---|---|
| export + graph rewrites + calibration (icefall container) | ~10 min |
| `pulsar2 build` | 34 min 29 s, peak RSS 9.55 GB |
| `pulsar2 run` verification | ~2 min |

```
cosine = 0.999132  MAE = 0.075091  shape = [1, 248, 512]
```

i.e. the CI-built axmodel matches the published one (cosine 0.999).

One caveat about the checkpoint: the 639 MB `checkpoint/pretrained.pt` in the GitHub
repo is numerically different from `streaming_exp/pretrained.pt` in the HF repo
(all 989 tensors differ). Quantizing the GitHub file with the settings in
`pulsar2-encoder.json` gives cosine ~0.972 (the text is still correct, but the
tensor-level match is worse), while the HF checkpoint gives 0.999, so
`prepare-axmodel.sh` downloads the HF one.

### Pipeline and expected timings

| Step | Where | Time (4-core runner) |
|---|---|---|
| Export fp32 static ONNX (Step 1) | icefall CPU container | ~10 min |
| `prepare-encoder.py` — If / shapes / Softplus | same container | ~5–10 min |
| `make-calibration-set.py` | same container | < 1 min |
| `pulsar2 build` (Step 4) | host, Pulsar2 8.0 | **~35 min** (measured on a 4-core GitHub runner, peak RSS 9.3 GB) |
| `pulsar2 run` verification (Step 5) | host | ~2 min |

Disk: ~15 GB free are needed (the icefall image ~7 GB can be pruned after the export step).
The measured run on `ubuntu-22.04` (4 cores, 16 GB RAM) took 10 min for export + rewrites and
34 min 26 s for `pulsar2 build`, peak RSS 9.27 GB, producing the same 165 MB `encoder.axmodel`.
As a sanity check of the toolchain itself, compiling the *joiner* takes ~17 s in the lite
package, so `pulsar2 build` is known to work on a plain Ubuntu 22.04 machine as well.

The calibration set in CI comes from the 4 test wavs used by
`../export-non-streaming.sh`, i.e. 4 samples. The released axmodel in
<https://huggingface.co/yunyu1258/x-asr-axmodel> used 8 real audio windows; passing more
wavs via `--wav-dir` gives a slightly better calibrated model.
