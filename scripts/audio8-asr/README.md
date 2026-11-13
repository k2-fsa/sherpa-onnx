# Audio8-ASR-0.1B → ONNX

This directory converts [`Edge0/Audio8-ASR-0.1B`](https://huggingface.co/Edge0/Audio8-ASR-0.1B)
into the two ONNX graphs that `sherpa-onnx` needs. The model is a Whisper-style
audio encoder followed by a small Qwen2 decoder, so it runs on CPU without any
AXERA/NPU specific code.

## What the model is

| part | description |
|---|---|
| features | Whisper log-mel, 128 bins, 25 ms window / 10 ms hop, padded to 3000 frames (30 s) |
| encoder | 3 stride-2 convolutions + 18 self-attention layers (d=896, 14 heads) + 4-layer MLP tower, output 1024 |
| audio tokens | 13 per 100 mel frames; the prompt uses `N = ((mel_frames + 1) / 2) / 4` of them |
| decoder | Qwen2, 8 layers, hidden 512, 8 heads × 64, RoPE theta 1e6, tied embeddings, vocab 151936 |
| prompt | `<\|user\|><\|begin_of_audio\|>` + `<\|audio\|>` × N + `<\|end_of_audio\|>Please transcribe this audio.<\|assistant\|>` |
| stop | 151645 (`<\|im_end\|>`); 151643 is the pad id |

The encoder and decoder are exported with **static shapes** (30 s window, 512 KV
cache slots) so they also run on NPUs later without a reshape pass.

## Graphs

`encoder.onnx`

```
inputs : audios    float32 [1, 128, 3000]   # log-mel, zero padded
         num_rows  int64   [1]              # encoder rows for the real audio length
outputs: hidden    float32 [1, 390, 1024]   # rows >= num_rows are invalid
```

`decoder.onnx`

```
inputs : input_ids       int64   [1, T]
         audio_hidden    float32 [1, K, 1024]   # pooled encoder rows
         attention_mask  int64   [1, 512]
         cache_position  int64   [T]
         cache_key_i / cache_value_i  float32 [1, 8, 512, 64], i = 0..7
outputs: logits          float32 [1, T, 151936]
         present_key_i / present_value_i  float32 [1, 8, T, 64]
```

The decoder applies the audio projector (LayerNorm + Linear 1024 → 512), injects
the projected rows at the `<|audio|>` positions and returns the KV slices for the
current step, which the runtime writes into the cache at `cache_position`.

`num_rows` exists because the HF audio encoder chunks the mel with a data
dependent `split()`, which cannot be traced. The exported graph always processes
30 chunks of 100 frames and masks the attention windows (104 tokens each) with
`num_rows`, which reproduces the variable-length graph exactly.

## Steps

```bash
# 0) dependencies
pip install torch transformers onnxruntime onnx numpy soundfile

# 1) download the checkpoint (the tokenizer files are needed as well)
huggingface-cli download Edge0/Audio8-ASR-0.1B --local-dir ./Audio8-ASR-0.1B

# 2) export the two graphs
python export-onnx.py --out-dir ./exported

# 3) sanity check: run the exported graphs and compare with transformers
python run-onnx.py ./wavs/0.wav
```

## Accuracy

Comparing the exported ONNX pipeline with `transformers` (fp32, eager attention,
greedy, `max_new_tokens=128`) on five test clips gives the **same text** on all of
them (character error rate 0.000):

| wav | duration | text |
|---|---|---|
| 0.wav | 10.05 s | 昨天是 Monday。Today is礼拜二。The day after tomorrow，是星期三。 |
| 1.wav | 5.10 s | 这是第一种。第二种叫呃，与os、os什么意思啊？ |
| 2.wav | 4.69 s | 就是平凡的啊，不认识接下来， frequent平凡的。 |
| 3.wav | 8.83 s | 第一句是个什么时代？加了ES是一般现代式，对，我们把它时代一些商品。 |
| 4.wav | 17.64 s | 嗯，On time 就准时，in time 是及时交。他总是准时交他的作业。… |

Tensor-level checks: encoder cosine ≈ 1.0, decoder prefill logits cosine ≈ 1.0
(max |diff| ≈ 4e-5), identical argmax.

### Two transformers 5.22 gotchas (both break the reference silently)

1. `language_model.lm_head.weight` is reported as MISSING and randomly
   initialised, because the release drops the tie (it was written with
   transformers 4.57.3). Re-tie it to `embed_tokens` before running.
2. The non-persistent `positional_embedding` buffer of
   `SinusoidsPositionEmbedding` is **zero-filled** by the loader (it is not in the
   checkpoint). Recompute the sinusoid table, otherwise the audio encoder silently
   loses positional information and the output degrades to short fragments.
   `export-onnx.py` restores it before tracing.

## Running with sherpa-onnx

```bash
# C++
./build/bin/sherpa-onnx-offline \
  --audio8-asr-encoder=./exported/encoder.onnx \
  --audio8-asr-decoder=./exported/decoder.onnx \
  --audio8-asr-tokenizer=./Audio8-ASR-0.1B \
  --num-threads=4 wavs/0.wav

# Python
python python-api-examples/offline-audio8-asr-decode-files.py \
  --encoder=./exported/encoder.onnx \
  --decoder=./exported/decoder.onnx \
  --tokenizer=./Audio8-ASR-0.1B \
  wavs/0.wav
```

Expected speed on an Apple M-series CPU (fp32, 4 threads): RTF ≈ 0.10, i.e. about
1 s for a 10 s clip.

## Notes

* The model is trained on 30 s windows, so longer audio has to be split, e.g. with
  a VAD. Audio beyond 30 s is truncated.
* The decoder cache is limited to `max_total_len = 512` slots (prompt + generated
  tokens). Raise `--audio8-asr-max-new-tokens` if a long sentence gets truncated.
* The license of the checkpoint is CC-BY-NC-4.0 — check it before commercial use.
