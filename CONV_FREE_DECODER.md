# Conv-free transducer decoder for GPU serving

The stateless decoder of icefall transducer models (zipformer and others with
`--context-size 2`) contains one grouped `Conv1d`. When sherpa-onnx runs that
decoder on the GPU with many concurrent streams, ONNX Runtime's CUDA `Conv`
kernel serializes all decoder runs behind a lock. icefall can now export the
decoder with that `Conv` replaced by an equivalent `MatMul`
([k2-fsa/icefall#2120](https://github.com/k2-fsa/icefall/pull/2120)). The
exported model has the same inputs, outputs and metadata, so sherpa-onnx loads
it with no code change.

On an A10G GPU serving 200-250 concurrent streams, the Conv-free decoder cut
the decoder's time per decode call from 80-116 ms to 8-10 ms, p95 latency by
22-28%, and server CPU by about one core. WER did not change.

## Who should use it

- You run a streaming or offline transducer model with `--provider=cuda` or
  `--provider=trt`. With `--provider=trt` only the encoder runs on TensorRT;
  the decoder and joiner run on ONNX Runtime's CUDA provider (see
  `GetSessionOptions(config, model_type)` in `sherpa-onnx/csrc/session.cc`).
  The lock affects both providers.
- Several threads decode at the same time with changing batch sizes. The
  streaming websocket server (`sherpa-onnx-online-websocket-server`) does
  this, and so does any application that calls `DecodeStreams` from several
  threads.
- The decoder has a `Conv` node (context size > 1). Models exported with
  `--context-size 1` have no decoder conv and are not affected.

CPU-only deployments are not affected: ONNX Runtime's CPU `Conv` takes no
lock. The Conv-free export has not been measured on CPU, so there is no reason
to switch for CPU-only use.

## Why the Conv serializes

ONNX Runtime's CUDA `Conv::ComputeInternal` (`onnxruntime/core/providers/cuda/nn/conv.cc`)
holds a per-node `std::mutex` for the whole call. Inside the lock it compares
the input shape with the single shape it cached from the previous run. If the
shape changed, it rebuilds the cuDNN execution plan, which takes about 2 ms.

A batching server changes the batch size on almost every call, and "the
previous run" is the previous run on any thread. So nearly every decoder run
pays the rebuild, one thread at a time. A standalone test with only the
decoder's `Conv` (random weights, one shared session, ONNX Runtime 1.28.2)
shows the effect:

| input batch sizes                     | Conv, 1 thread | Conv, 32 threads | MatMul, 1 thread | MatMul, 32 threads |
|---------------------------------------|---------------:|-----------------:|-----------------:|-------------------:|
| constant: 100                         |          8,919 |           14,316 |            8,256 |              9,168 |
| alternating: 100, 200                 |            969 |              998 |            6,234 |              6,203 |
| alternating: 40, 100, 200, 400        |            654 |              655 |            5,477 |              3,739 |

(runs per second)

With a constant shape the `Conv` is fast. As soon as the shape varies, it is
capped at about 650-1,000 runs/s regardless of the number of threads.

## The rewrite

At inference the decoder conv sees exactly `context_size` frames (`need_pad=False`)
and produces one frame. A grouped conv with one output frame is a
block-diagonal linear map of the flattened input:

```
y[n, co, 0] = sum_{i,k} W[co, i, k] * x[n, g*cin_g + i, k],   g = co // cout_g
            = flatten(x)[n, :] @ W_dense[:, co]
W_dense[(g*cin_g + i)*kernel_size + k, co] = W[co, i, k]
```

The exported graph replaces

```
Conv(x [N, 512, 2], weight [512, 4, 2], group=128)  ->  [N, 512, 1]
```

with

```
Reshape(x, [N, 1024]) -> MatMul(., W_dense [1024, 512]) -> Unsqueeze(axis=2)  ->  [N, 512, 1]
```

Every other node is unchanged. For a 512-dim decoder the file grows by about
2 MB (for example 3.17 MB to 5.25 MB), because the dense weight stores the
zeros of the block-diagonal matrix. The extra multiply-adds are negligible on
the GPU: the decoder run gets faster, not slower.

## Export

Add `--decoder-conv-as-matmul 1` to `zipformer/export-onnx-streaming.py` or
`zipformer/export-onnx.py` in icefall (any recipe that uses the zipformer
export scripts). All other arguments stay the same as for your normal export:

```bash
cd icefall/egs/librispeech/ASR

./zipformer/export-onnx-streaming.py \
  --tokens $repo/data/lang_bpe_500/tokens.txt \
  --use-averaged-model 0 \
  --epoch 99 \
  --avg 1 \
  --exp-dir $repo/exp \
  --num-encoder-layers "2,2,3,4,3,2" \
  --downsampling-factor "1,2,4,8,4,2" \
  --feedforward-dim "512,768,1024,1536,1024,768" \
  --num-heads "4,4,4,8,4,4" \
  --encoder-dim "192,256,384,512,384,256" \
  --query-head-dim 32 \
  --value-head-dim 12 \
  --pos-head-dim 4 \
  --pos-dim 48 \
  --encoder-unmasked-dim "192,192,256,256,256,192" \
  --cnn-module-kernel "31,31,15,15,15,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --causal True \
  --chunk-size 16 \
  --left-context-frames 128 \
  --decoder-conv-as-matmul 1
```

The option is off by default. When it is on, the script also exports the
original Conv decoder to a temporary file, runs both on the same random token
ids with onnxruntime (GPU if available, TF32 off; else CPU) and fails if they
differ by more than 1e-4. It logs the result, for example:

```
Checked the decoder with MatMul against the decoder with Conv (onnxruntime on cuda): max abs diff 9.54e-07
```

The encoder and joiner are exported exactly as before. Only the decoder file
differs.

To confirm that an exported decoder has no `Conv`:

```python
import onnx

m = onnx.load("decoder-epoch-99-avg-1-chunk-16-left-128.onnx")
print(sorted({n.op_type for n in m.graph.node}))
# Conv-free: no 'Conv' in the list; 'MatMul' and 'Reshape' are present.
```

## Use in sherpa-onnx

Pass the new file as the decoder. Nothing else changes, in any API or
language binding:

```bash
./build/bin/sherpa-onnx-online-websocket-server \
  --tokens=./tokens.txt \
  --encoder=./encoder.onnx \
  --decoder=./decoder-conv-as-matmul.onnx \
  --joiner=./joiner.onnx \
  --provider=cuda \
  --decoding-method=modified_beam_search \
  --max-active-paths=10
```

To roll back, pass the original `decoder.onnx` again.

## Results

Setup: 1x NVIDIA A10G, 8 vCPU; ONNX Runtime 1.28.2 with the CUDA provider,
TensorRT 10.13, CUDA 12.9, cuDNN 9.10; a streaming zipformer2 English model at
8 kHz, FP32; `modified_beam_search` with 10 active paths. The Conv-free decoder
in these runs was produced from the same weights by the same graph rewrite
(Reshape + MatMul + Unsqueeze). It matched the original decoder to within
1e-6 for batch sizes 1, 7, 100 and 500.

### Decoder alone

One shared decoder session and one shared joiner session (no sherpa-onnx). T
threads repeat the server's per-frame sequence (decoder run, then joiner run)
with batch sizes drawn from {40, 100, 200, 400}. CUDA provider, heuristic cuDNN
algorithm search, as in sherpa-onnx.

| threads | original: runs/s | p50 ms/run | Conv-free: runs/s | p50 ms/run | speed-up |
|--------:|-----------------:|-----------:|------------------:|-----------:|---------:|
|       1 |              511 |       2.16 |             2,491 |       0.16 |     4.9x |
|       8 |              588 |      13.14 |             4,383 |       0.76 |     7.5x |
|      32 |              594 |      53.20 |             4,269 |       0.98 |     7.2x |
|      64 |              588 |     108.65 |             4,119 |       2.35 |     7.0x |

The original decoder stays at about 590 runs/s from 8 to 64 threads, and its
per-run latency grows with the number of waiting threads: one serial section.
With 32 threads, 88% of all thread time was spent blocked on CUDA-provider
mutexes. With the Conv-free decoder that wait almost disappears.

### Streaming server

`sherpa-onnx-online-websocket-server` with the TensorRT encoder, decoder and
joiner on the CUDA provider, `--loop-interval-ms=50`, `--max-batch-size=50`,
100 work / I/O / intra-op threads. The load was a replicated 500-utterance set
at fixed concurrency. Latencies are per-chunk server latency in ms (middle 80%
of each run). Each number is from a single run, so treat differences under
about 10% as noise.

|                                 | original, 200 streams | Conv-free, 200 streams | original, 250 streams | Conv-free, 250 streams |
|---------------------------------|----------------------:|-----------------------:|----------------------:|-----------------------:|
| p50 latency (ms)                |                 244.1 |                  164.5 |                 316.3 |                  200.7 |
| p95 latency (ms)                |                 328.9 |                  258.0 |                 456.9 |                  330.3 |
| p99 latency (ms)                |                 394.2 |                  313.8 |                 631.6 |                  381.2 |
| decoder time per call (ms)      |                  80.4 |                    8.2 |                 115.5 |                   10.4 |
| decoder CPU per call (ms)       |                  50.1 |                    6.4 |                  53.9 |                    7.4 |
| `DecodeStreams` wall time (ms)  |                 185.8 |                  103.7 |                 247.9 |                  135.2 |
| server CPU (cores)              |                   3.0 |                    2.0 |                   3.6 |                    2.6 |
| GPU utilization (%)             |                  32.4 |                   32.9 |                  35.2 |                   36.5 |
| WER (%)                         |                   8.3 |                    8.3 |                   8.3 |                    8.3 |

- p95 latency is 22% lower at 200 streams and 28% lower at 250. p99 is 20% and
  40% lower.
- The decoder's time per decode call falls by about 10x, and the server needs
  about one CPU core less.
- GPU utilization and memory are unchanged. The GPU was not the bottleneck;
  the lock was.
- WER is identical.

With the decoder lock gone, the next costs in a decode call are the CPU
log-softmax in `modified_beam_search` and the encoder run. The Conv-free
decoder is also the first step for further GPU-side optimizations: binding the
decoder output to the GPU (I/O binding) gave a further 21-54% decoder
throughput in the isolated test, but only once the `Conv` lock was gone (+1-5%
with it).

## Alternatives

- **Run the original decoder on TensorRT.** In the isolated test the original
  decoder on the TensorRT provider scaled like the Conv-free one (4,094 vs
  4,254 runs/s at 32 threads), but with a higher median latency per run
  (2.49 ms vs 1.03 ms). It also needs a TensorRT engine and shape profile for
  the decoder. sherpa-onnx does not run the decoder on TensorRT today, and
  this was not tested in the server.
- **Pad the decoder batch to a fixed size.** With a constant shape the `Conv`
  does not rebuild its plan. Even two alternating sizes already cost about 9x,
  so this would need one fixed size for every call. Not tested.
