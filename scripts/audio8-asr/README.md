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
#    export-onnx.py needs torch + transformers; run-onnx.py only needs
#    onnxruntime, kaldi-native-fbank, librosa and tokenizers (the same
#    feature/tokenizer stack as the C++ runtime, no transformers)
pip install torch transformers onnx onnxruntime numpy librosa \
    kaldi-native-fbank tokenizers

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

## Released model directories

The GitHub Actions job (`workflow_dispatch`) does not only export the graphs, it
also assembles two ready-to-use directories and publishes them:

```
sherpa-onnx-audio8-asr-0.1b/
  encoder.onnx  decoder.onnx  LICENSE  README.md
  tokenizer/    test_wavs/
sherpa-onnx-audio8-asr-0.1b-int8/
  encoder.int8.onnx  decoder.int8.onnx  LICENSE  README.md
  tokenizer/         test_wavs/
```

`test_wavs/` holds one clip for each supported language (`en`, `zh`, `fr`, `de`,
`ja`, `ko`, `cantonese`). Both directories are packed into `.tar.bz2` files,
attached to the `asr-models` release and pushed to
<https://huggingface.co/csukuangfj/audio8-asr-0.1b> and
<https://huggingface.co/csukuangfj/audio8-asr-0.1b-int8> (the push steps are
`continue-on-error` so the job still passes without an `HF_TOKEN0` secret).

`encoder.onnx` / `decoder.onnx` (and the int8 copies) carry metadata describing
the model (`model_type`, `url`, `license`, `version`, `maintainer`, plus the
tensor dimensions), so tools can identify them without a side-car file.

## Running with sherpa-onnx

Prebuilt ONNX models (fp32 and dynamic-int8) are on HuggingFace:
<https://huggingface.co/yunyu1258/audio8-asr-onnx> — `encoder.onnx`,
`decoder.onnx`, `encoder.int8.onnx`, `decoder.int8.onnx`, `tokenizer/`,
`test_wavs/`. They can also be reproduced with
`.github/workflows/export-audio8-asr-onnx.yaml`.

Run the commands below **from the repository root** (replace the paths with the
place where you put the model):

```bash
# C++
./build/bin/sherpa-onnx-offline \
  --audio8-asr-encoder=./Audio8-ASR-0.1B/encoder.onnx \
  --audio8-asr-decoder=./Audio8-ASR-0.1B/decoder.onnx \
  --audio8-asr-tokenizer=./Audio8-ASR-0.1B/tokenizer \
  --num-threads=4 ./Audio8-ASR-0.1B/test_wavs/0.wav

# Python (`pip install sherpa-onnx`, or use the package built from this tree)
python3 python-api-examples/offline-audio8-asr-decode-files.py \
  --encoder=./Audio8-ASR-0.1B/encoder.onnx \
  --decoder=./Audio8-ASR-0.1B/decoder.onnx \
  --tokenizer=./Audio8-ASR-0.1B/tokenizer \
  ./Audio8-ASR-0.1B/test_wavs/0.wav
```

For lower CPU cost use `encoder.int8.onnx` / `decoder.int8.onnx` with the same
flags; on the five test clips the transcripts are identical to fp32.

Expected speed on an Apple M-series CPU (fp32, 4 threads): RTF ≈ 0.1, i.e. about
1 s for a 10 s clip; the int8 graphs are about 2x faster.

## Notes

* The model is trained on 30 s windows, so longer audio has to be split, e.g. with
  a VAD. Audio beyond 30 s is truncated.
* The decoder cache has 512 slots (prompt + generated tokens); the size is read
  from the exported decoder, not configured. Raise `--audio8-asr-max-new-tokens`
  (or `stream->SetOption("max_new_tokens", "...")`) if a long sentence gets
  truncated, and shorten the audio when prompt + generation does not fit.
* The license of the checkpoint is CC-BY-NC-4.0 — check it before commercial use.
