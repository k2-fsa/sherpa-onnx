# Introduction

This folder uses Rust API maintained by us.

## Setup library path

### Method 1 (Build from source, support only shared libs right now)

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
| 4 | [zipvoice_tts](#example-4-tts-with-zipvoice-zero-shot-voice-cloning) | Text-to-speech with ZipVoice zero-shot voice cloning |
| 5 | [vits_tts](#example-5-tts-with-vits-english-piper) | Text-to-speech with a standalone VITS Piper model (English) |
| 6 | [vits_tts](#example-6-tts-with-vits-german-piper) | Text-to-speech with a standalone VITS Piper model (German) |
| 7 | [matcha_tts_en](#example-7-tts-with-matcha-english) | Text-to-speech with Matcha TTS (English) |
| 8 | [matcha_tts_zh](#example-8-tts-with-matcha-chinese) | Text-to-speech with Matcha TTS (Chinese) |
| 9 | [kokoro_tts_en](#example-9-tts-with-kokoro-english) | Text-to-speech with Kokoro TTS (English) |
| 10 | [kokoro_tts_zh_en](#example-10-tts-with-kokoro-chinese--english) | Text-to-speech with Kokoro TTS (Chinese + English) |
| 11 | [kitten_tts_en](#example-11-tts-with-kitten-english) | Text-to-speech with Kitten TTS (English) |
| 12 | [streaming_zipformer_en](#example-12-asr-with-streaming-zipformer-english) | Streaming ASR with zipformer transducer (English) |
| 13 | [streaming_zipformer_zh_en](#example-13-asr-with-streaming-zipformer-chinese--english) | Streaming ASR with zipformer transducer (Chinese + English) |
| 14 | [streaming_zipformer_microphone](#example-14-asr-with-streaming-zipformer-with-a-microphone-real-time-asr) | Real-time streaming ASR from microphone input |
| 15 | [zipformer_en](#example-15-asr-with-non-streaming-zipformer-english) | Non-streaming ASR with zipformer transducer (English) |
| 16 | [zipformer_zh_en](#example-16-asr-with-non-streaming-zipformer-chinese--english) | Non-streaming ASR with zipformer transducer (Chinese + English) |
| 17 | [zipformer_vi](#example-17-asr-with-non-streaming-zipformer-vietnamese) | Non-streaming ASR with zipformer transducer (Vietnamese) |
| 18 | [nemo_parakeet](#example-18-asr-with-non-streaming-nemo-parakeet-english) | Non-streaming ASR with Nemo Parakeet TDT transducer (English) |
| 19 | [fire_red_asr_ctc](#example-19-asr-with-non-streaming-fireredasr-ctc-chinese--english) | Non-streaming ASR with FireRedASR CTC model (Chinese + English) |
| 20 | [moonshine_v2](#example-20-asr-with-non-streaming-moonshine-v2-english) | Non-streaming ASR with Moonshine v2 (English) |
| 21 | [sense_voice](#example-21-asr-with-non-streaming-sensevoice) | Non-streaming ASR with SenseVoice (Chinese, English, Japanese, Korean, Cantonese) |
| 22 | [silero_vad_remove_silence](#example-22-remove-silences-from-a-file-using-silerovad) | Remove silences from an audio file using Silero VAD |
| 23 | [offline_speech_enhancement_gtcrn](#example-23-offline-speech-enhancement-with-gtcrn) | Offline speech enhancement with GTCRN |
| 24 | [offline_speech_enhancement_dpdfnet](#example-24-offline-speech-enhancement-with-dpdfnet) | Offline speech enhancement with DPDFNet |
| 25 | [streaming_speech_enhancement_gtcrn](#example-25-streaming-speech-enhancement-with-gtcrn) | Streaming speech enhancement with GTCRN |
| 26 | [streaming_speech_enhancement_dpdfnet](#example-26-streaming-speech-enhancement-with-dpdfnet) | Streaming speech enhancement with DPDFNet |
| 27 | [online_punctuation](#example-27-online-punctuation) | Add punctuation to text using online punctuation model |

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

### Example 4: TTS with ZipVoice zero-shot voice cloning

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
tar xvf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
rm sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

cargo run --example zipvoice_tts
```


### Example 5: TTS with VITS (English Piper)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2
rm vits-piper-en_US-amy-low.tar.bz2

cargo run --example vits_tts -- \
  --model ./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
  --tokens ./vits-piper-en_US-amy-low/tokens.txt \
  --data-dir ./vits-piper-en_US-amy-low/espeak-ng-data \
  --output ./generated-vits-en-rust.wav \
  --text "Liliana, the most beautiful and lovely assistant of our team!"
```

### Example 6: TTS with VITS (German Piper)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-glados-high.tar.bz2
tar xf vits-piper-de_DE-glados-high.tar.bz2
rm vits-piper-de_DE-glados-high.tar.bz2

cargo run --example vits_tts -- \
  --model ./vits-piper-de_DE-glados-high/de_DE-glados-high.onnx \
  --tokens ./vits-piper-de_DE-glados-high/tokens.txt \
  --data-dir ./vits-piper-de_DE-glados-high/espeak-ng-data \
  --output ./generated-vits-de-rust.wav \
  --text "Alles hat ein Ende, nur die Wurst hat zwei."
```

### Example 7: TTS with Matcha (English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

cargo run --example matcha_tts_en
```

### Example 8: TTS with Matcha (Chinese)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

cargo run --example matcha_tts_zh
```

### Example 9: TTS with Kokoro (English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

cargo run --example kokoro_tts_en
```

### Example 10: TTS with Kokoro (Chinese + English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
tar xf kokoro-multi-lang-v1_0.tar.bz2
rm kokoro-multi-lang-v1_0.tar.bz2

cargo run --example kokoro_tts_zh_en
```

### Example 11: TTS with Kitten (English)

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
tar xf kitten-nano-en-v0_1-fp16.tar.bz2
rm kitten-nano-en-v0_1-fp16.tar.bz2

cargo run --example kitten_tts_en
```

### Example 12: ASR with streaming zipformer (English)

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

### Example 13: ASR with streaming zipformer (Chinese + English)

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

### Example 14: ASR with streaming zipformer (with a microphone, real-time ASR)

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

### Example 15: ASR with non-streaming zipformer (English)

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

### Example 16: ASR with non-streaming zipformer (Chinese + English)

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

### Example 17: ASR with non-streaming zipformer (Vietnamese)

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

### Example 18: ASR with non-streaming Nemo Parakeet (English)

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

### Example 19: ASR with non-streaming FireRedASR CTC (Chinese + English)

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

### Example 20: ASR with non-streaming Moonshine v2 (English)

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

### Example 21: ASR with non-streaming SenseVoice

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

### Example 22: Remove silences from a file using SileroVAD

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

cargo run --example silero_vad_remove_silence -- \
    --input ./lei-jun-test.wav \
    --output ./no-silence.wav \
    --silero-vad-model ./silero_vad.onnx
```

### Example 23: Offline speech enhancement with GTCRN

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

cargo run --example offline_speech_enhancement_gtcrn -- \
    --model ./gtcrn_simple.onnx \
    --input ./inp_16k.wav \
    --output ./enhanced-rust-gtcrn.wav
```

### Example 24: Offline speech enhancement with DPDFNet

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

cargo run --example offline_speech_enhancement_dpdfnet -- \
    --model ./dpdfnet_baseline.onnx \
    --input ./inp_16k.wav \
    --output ./enhanced-rust-dpdfnet.wav
```

### Example 25: Streaming speech enhancement with GTCRN

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

cargo run --example streaming_speech_enhancement_gtcrn -- \
    --model ./gtcrn_simple.onnx \
    --input ./inp_16k.wav \
    --output ./enhanced-rust-streaming-gtcrn.wav
```

### Example 26: Streaming speech enhancement with DPDFNet

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

cargo run --example streaming_speech_enhancement_dpdfnet -- \
    --model ./dpdfnet_baseline.onnx \
    --input ./inp_16k.wav \
    --output ./enhanced-rust-streaming-dpdfnet.wav
```

### Example 27: Online punctuation

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2

cargo run --example online_punctuation -- \
    --cnn-bilstm ./sherpa-onnx-online-punct-en-2024-08-06/model.onnx \
    --bpe-vocab ./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab
```
