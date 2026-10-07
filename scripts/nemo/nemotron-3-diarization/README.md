# Nemotron-3-Diarization

This directory exports
[nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
to ONNX for end-to-end speaker diarization with sherpa-onnx.

The model is a streaming Sortformer with up to 8 speakers. Speakers are
numbered in the order of their first arrival. It needs neither a speaker
embedding model nor clustering.

## Export

The export uses the PyTorch implementation from Hugging Face transformers.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/huggingface/transformers"
pip install onnx onnxruntime onnxscript librosa soundfile

./run.sh
```

It writes `model.onnx` (about 400 MB) and `model.int8.onnx` (about 100 MB).

## How the exported model is used

`model.onnx` is stateless. One call processes one chunk of the audio:

| Input | Shape | Description |
|-------|-------|-------------|
| `features` | `(N, T, 128)` | Log-mel frames of the chunk and its right context. `T` must be a multiple of 8 |
| `cached_embeds` | `(N, C, 512)` | Speaker cache followed by the FIFO queue. `C` can be 0 |

| Output | Shape | Description |
|--------|-------|-------------|
| `probs` | `(N, (C + T/8) * 8, 8)` | Speaker activity probabilities, one row per 10 ms |
| `chunk_embeds` | `(N, T/8, 512)` | Embeddings of the chunk, to push to the FIFO queue |

The caller keeps the Arrival-Order Speaker Cache (AOSC) and the FIFO queue
between the chunks.
[./test_onnx.py](./test_onnx.py) implements it with numpy, and
[sortformer-speaker-cache.cc](../../../sherpa-onnx/csrc/sortformer-speaker-cache.cc)
implements it in C++. The model metadata holds the frontend parameters, the
chunk geometry and the speaker cache policy.

The features are NeMo log-mel features without normalization: preemphasis
0.97, a 400-sample symmetric Hann window, a 512-point FFT, a 160-sample hop,
128 librosa (Slaney) mel bins and `log(x + 2**-24)`.

The default chunking is the offline configuration of the model card:
chunks of 340 encoder frames (27.2 s) with 40 frames of right context, a
FIFO queue of 40 frames, a speaker cache of 264 frames and an update period
of 300 frames.

## Accuracy

`./test_onnx.py --reference` compares the ONNX model and the numpy speaker
cache with the transformers implementation. On a 227 s recording made of two
meeting clips, the largest difference of the speaker probabilities is 1.3e-5. The
low-latency configuration (`--streaming`, 1.04 s chunks) gives the same
result. The last 80 ms of a recording can differ when the number of 10 ms
frames is a multiple of 8, as the reference then attends to one more,
zero-padded, encoder frame.
