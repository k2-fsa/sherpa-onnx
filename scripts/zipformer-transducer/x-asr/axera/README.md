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
| 99 `Log(1 + Exp(-|x|))` patterns (the `SwooshL/SwooshR` activations) | Pulsar2 has no `Log`; the graph is split into 17 NPU + 16 ONNX subgraphs, which is slow to compile and slow at runtime | `log(1 + exp(z)) == softplus(z)`, so replace them by `Softplus` (which Pulsar2 supports) → a single NPU subgraph |

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
sherpa-onnx model.

## Step 2. Prepare the graph for Pulsar2

```bash
python3 prepare-encoder.py \
  --src encoder-epoch-99-avg-1.onnx \
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

Compare the axmodel output with the ONNX output for the same input. With Pulsar2's
simulator:

```bash
pulsar2 run --model output_encoder/encoder.axmodel \
    --input_dir ./sim_input --output_dir ./sim_output
```

`sim_input/x4d.bin` is a raw float32 dump of one calibration sample, and the
comparison is done against the ONNX reference output.

Reference numbers obtained with this pipeline (AX650 simulation, 4 test wavs):

| Metric | Value |
|---|---|
| cosine (mean / min) | 0.9981 / 0.9967 |
| MAE | 0.08 – 0.14 |
| greedy-search text | 1 of 4 utterances token-identical to fp32; the rest differ in a few English words |

> Note: `pulsar2 run` compares its output against any `<output>.bin` that already
> exists in `--output_dir` and asserts bit-exact equality. Use a fresh directory
> for every sample.

## Deployment

The prebuilt models are published on HuggingFace
(<https://huggingface.co/yunyu1258/x-asr-axmodel>) together with a Python example
(`run_asr.py`, uses `axengine` + `numpy` only).
