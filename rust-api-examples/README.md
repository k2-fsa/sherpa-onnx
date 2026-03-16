# Introduction

This folder uses Rust API maintained by us.

## Setup library path

### Method 1 (Build from source)

```bash
export SHERPA_ONNX_LIB_DIR=/Users/fangjun/open-source/sherpa-onnx/build/install/lib
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$SHERPA_ONNX_LIB_DIR"
```

### Method 2 (Download pre-built libs)

```bash
# You can choose any directory you like
cd $HOME/Downloads

# We use version v1.12.25 below as an example.
# Please always use the latest version from
# https://github.com/k2-fsa/sherpa-onnx/releases

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.25/sherpa-onnx-v1.12.25-osx-universal2-shared.tar.bz2
tar xvf sherpa-onnx-v1.12.25-osx-universal2-shared.tar.bz2
rm sherpa-onnx-v1.12.25-osx-universal2-shared.tar.bz2

export SHERPA_ONNX_LIB_DIR=$HOME/Downloads/sherpa-onnx-v1.12.25-osx-universal2-shared/lib
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$SHERPA_ONNX_LIB_DIR"
```

## Examples

| # | Example | Description |
|---|---------|-------------|
| 1 | [version](#example-1-show-sherpa-onnx-version) | Show the sherpa-onnx version |
| 2 | [pocket_tts](#example-2-tts-with-pocket-tts-zero-shot-voice-cloning) | Text-to-speech with zero-shot voice cloning using a reference audio |
| 3 | [supertonic_tts](#example-3-tts-with-supertonic-tts) | Text-to-speech with Supertonic TTS (multi-speaker, multi-language) |
| 4 | [streaming_zipformer_en](#example-4-asr-with-streaming-zipformer-english) | Streaming ASR with zipformer transducer (English) |
| 5 | [streaming_zipformer_zh_en](#example-5-asr-with-streaming-zipformer-chinese--english) | Streaming ASR with zipformer transducer (Chinese + English) |
| 6 | [streaming_zipformer_microphone](#example-6-asr-with-streaming-zipformer-with-a-microphone-real-time-asr) | Real-time streaming ASR from microphone input |
| 7 | [zipformer_en](#example-7-asr-with-non-streaming-zipformer-english) | Non-streaming ASR with zipformer transducer (English) |
| 8 | [zipformer_zh_en](#example-8-asr-with-non-streaming-zipformer-chinese--english) | Non-streaming ASR with zipformer transducer (Chinese + English) |
| 9 | [zipformer_vi](#example-9-asr-with-non-streaming-zipformer-vietnamese) | Non-streaming ASR with zipformer transducer (Vietnamese) |
| 10 | [nemo_parakeet](#example-10-asr-with-non-streaming-nemo-parakeet-english) | Non-streaming ASR with Nemo Parakeet TDT transducer (English) |
| 11 | [fire_red_asr_ctc](#example-11-asr-with-non-streaming-fireredasr-ctc-chinese--english) | Non-streaming ASR with FireRedASR CTC model (Chinese + English) |
| 12 | [moonshine_v2](#example-12-asr-with-non-streaming-moonshine-v2-english) | Non-streaming ASR with Moonshine v2 (English) |
| 13 | [sense_voice](#example-13-asr-with-non-streaming-sensevoice) | Non-streaming ASR with SenseVoice (Chinese, English, Japanese, Korean, Cantonese) |
| 14 | [silero_vad_remove_silence](#example-14-remove-silences-from-a-file-using-silerovad) | Remove silences from an audio file using Silero VAD |

## Run it

### Example 1: Show sherpa-onnx version

```bash
cargo run --example version
```

For macOS, you can run
```
otool -l target/debug/examples/version | grep -A2 LC_RPATH
```
to check the RPATH.

### Example 2: TTS with Pocket TTS (zero-shot voice cloning)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xvf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
rm sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

cargo run --example pocket_tts
```

### Example 3: TTS with Supertonic TTS

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
tar xvf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
rm sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2

cargo run --example supertonic_tts
```

### Example 4: ASR with streaming zipformer (English)

```bash
curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2
rm sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2

cargo run --example streaming_zipformer -- \
    --wav sherpa-onnx-streaming-zipformer-en-2023-06-21/test_wavs/1.wav \
    --encoder sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.int8.onnx \
    --decoder sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx \
    --joiner sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.int8.onnx \
    --tokens sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt \
    --provider cpu \
    --debug
```

### Example 5: ASR with streaming zipformer (Chinese + English)

```bash
curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

cargo run --example streaming_zipformer -- \
    --wav sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/2.wav \
    --encoder sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx \
    --decoder sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    --joiner sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx \
    --tokens sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    --provider cpu \
    --debug
```

### Example 6: ASR with streaming zipformer (with a microphone, real-time ASR)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

cargo run --example streaming_zipformer_microphone --features mic -- \
    --encoder sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx \
    --decoder sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
    --joiner sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx \
    --tokens sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
    --provider cpu \
    --debug
```

### Example 7: ASR with non-streaming zipformer (English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04.tar.bz2
tar xvf icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04.tar.bz2
rm icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04.tar.bz2

cargo run --example zipformer -- \
    --wav "./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/test_wavs/1089-134686-0001.wav" \
    --tokens ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/data/lang_bpe_500/tokens.txt \
    --encoder ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/encoder-epoch-30-avg-4.int8.onnx \
    --decoder ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/decoder-epoch-30-avg-4.onnx \
    --joiner ./icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04/exp/joiner-epoch-30-avg-4.int8.onnx \
    --provider cpu \
    --num-threads 2 \
    --debug
```

### Example 8: ASR with non-streaming zipformer (Chinese + English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
tar xvf sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
rm sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2

cargo run --example zipformer -- \
    --wav "./sherpa-onnx-zipformer-zh-en-2023-11-22/test_wavs/0.wav" \
    --encoder "./sherpa-onnx-zipformer-zh-en-2023-11-22/encoder-epoch-34-avg-19.int8.onnx" \
    --decoder "./sherpa-onnx-zipformer-zh-en-2023-11-22/decoder-epoch-34-avg-19.onnx" \
    --joiner "./sherpa-onnx-zipformer-zh-en-2023-11-22/joiner-epoch-34-avg-19.int8.onnx" \
    --tokens "./sherpa-onnx-zipformer-zh-en-2023-11-22/tokens.txt" \
    --provider cpu \
    --num-threads 2 \
    --debug
```

### Example 9: ASR with non-streaming zipformer (Vietnamese)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2
tar xvf sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2
rm sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2

cargo run --example zipformer -- \
    --wav "./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/test_wavs/0.wav" \
    --encoder "./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/encoder.int8.onnx" \
    --decoder "./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/decoder.onnx" \
    --joiner "./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/joiner.int8.onnx" \
    --tokens "./sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/tokens.txt" \
    --provider cpu \
    --num-threads 2 \
    --debug
```

### Example 10: ASR with non-streaming Nemo Parakeet (English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2

cargo run --example nemo_parakeet -- \
    --wav "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav" \
    --encoder "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx" \
    --decoder "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx" \
    --joiner "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx" \
    --tokens "./sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt" \
    --provider cpu \
    --num-threads 2 \
    --debug
```

### Example 11: ASR with non-streaming FireRedASR CTC (Chinese + English)

```bash
curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
tar xvf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2
rm sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2

cargo run --example fire_red_asr_ctc -- \
    --wav ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/1.wav \
    --model ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx \
    --tokens ./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt \
    --num-threads 2 \
    --debug
```

### Example 12: ASR with non-streaming Moonshine v2 (English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2

cargo run --example moonshine_v2 -- \
    --wav ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav \
    --encoder ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort \
    --decoder ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort \
    --tokens ./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt \
    --num-threads 2
```

### Example 13: ASR with non-streaming SenseVoice

```bash
curl -SsL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

cargo run --example sense_voice -- \
    --wav ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/en.wav \
    --model ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx \
    --tokens ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
    --num-threads 2 \
    --debug
```

### Example 14: Remove silences from a file using SileroVAD

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

cargo run --example silero_vad_remove_silence -- \
    --input ./lei-jun-test.wav \
    --output ./no-silence.wav \
    --silero-vad-model ./silero_vad.onnx
```
