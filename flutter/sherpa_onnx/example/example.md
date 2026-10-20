# sherpa-onnx app example

## Contents

- [Pre-built Flutter apps](#pre-built-flutter-apps)
  - [VAD from file](#vad-from-file)
  - [VAD from microphone](#vad-from-microphone)
  - [VAD + ASR from file](#vad--asr-from-file)
    - [Zipformer CTC (Chinese)](#zipformer-ctc-chinese)
    - [SenseVoice (Chinese/English/Japanese/Korean/Cantonese)](#sensevoice-chineseenglishjapanesekoreancantonese)
    - [Whisper tiny.en (English)](#whisper-tinyen-english)
    - [NeMo Parakeet TDT v2 (English)](#nemo-parakeet-tdt-v2-english)
    - [NeMo Parakeet TDT v3 (25 European languages)](#nemo-parakeet-tdt-v3-25-european-languages)
    - [Moonshine tiny (English)](#moonshine-tiny-english)
    - [Qwen3 ASR 0.6B (multilingual)](#qwen3-asr-06b-multilingual)
    - [FunASR Nano (Chinese/English/Japanese)](#funasr-nano-chineseenglishjapanese)
    - [FireRed ASR CTC v2 (Chinese/English)](#firered-asr-ctc-v2-chineseenglish)
  - [VAD + ASR from microphone](#vad--asr-from-microphone)
    - [Zipformer CTC (Chinese)](#zipformer-ctc-chinese-1)
    - [SenseVoice (Chinese/English/Japanese/Korean/Cantonese)](#sensevoice-chineseenglishjapanesekoreancantonese-1)
    - [Whisper tiny.en (English)](#whisper-tinyen-english-1)
    - [NeMo Parakeet TDT v2 (English)](#nemo-parakeet-tdt-v2-english-1)
    - [NeMo Parakeet TDT v3 (25 European languages)](#nemo-parakeet-tdt-v3-25-european-languages-1)
    - [Moonshine tiny (English)](#moonshine-tiny-english-1)
    - [Qwen3 ASR 0.6B (multilingual)](#qwen3-asr-06b-multilingual-1)
    - [FunASR Nano (Chinese/English/Japanese)](#funasr-nano-chineseenglishjapanese-1)
    - [FireRed ASR CTC v2 (Chinese/English)](#firered-asr-ctc-v2-chineseenglish-1)
  - [Streaming speech recognition](#streaming-speech-recognition)
    - [Streaming Zipformer Bilingual (Chinese/English)](#streaming-zipformer-bilingual-chineseenglish)
    - [Streaming Zipformer (English)](#streaming-zipformer-english)
    - [Streaming Zipformer WenetSpeech (Chinese)](#streaming-zipformer-wenetspeech-chinese)
    - [Streaming Zipformer (French)](#streaming-zipformer-french)
    - [X-ASR 960ms (Chinese/English, with punctuation)](#x-asr-960ms-chineseenglish-with-punctuation)
    - [Nemotron 3.5 ASR Streaming 560ms (multilingual)](#nemotron-35-asr-streaming-560ms-multilingual)
    - [X-ASR 480ms (Chinese/English, with punctuation)](#x-asr-480ms-chineseenglish-with-punctuation)
    - [Nemotron 3.5 ASR Streaming 1120ms (multilingual)](#nemotron-35-asr-streaming-1120ms-multilingual)
  - [Text to speech](#text-to-speech)
    - [VITS Piper en_US-amy-low (English)](#vits-piper-en_us-amy-low-english)
    - [VITS Piper zh_CN-xiao_ya-medium (Chinese)](#vits-piper-zh_cn-xiao_ya-medium-chinese)
    - [VITS Piper en_US-libritts_r-medium (English)](#vits-piper-en_us-libritts_r-medium-english)
    - [VITS Inflect en-nano-v2 (English)](#vits-inflect-en-nano-v2-english)
    - [Kokoro v0.19 (English)](#kokoro-v019-english)
    - [Kokoro v1.0 (Chinese/English)](#kokoro-v10-chineseenglish)
    - [Matcha en_US-ljspeech (English)](#matcha-en_us-ljspeech-english)
    - [Matcha zh-en (Chinese/English)](#matcha-zh-en-chineseenglish)
    - [Kitten Nano v0.1 (English)](#kitten-nano-v01-english)
    - [PocketTTS (multilingual)](#pockettts-multilingual)
    - [Supertonic 3 (multilingual)](#supertonic-3-multilingual)
  - [Offline punctuation restoration](#offline-punctuation-restoration)
  - [Online punctuation restoration](#online-punctuation-restoration)
- [Web demos](#web-demos)
- [Flutter source code](#flutter-source-code)
- [Pure dart-examples](#pure-dart-examples)

## Pre-built Flutter apps

Download pre-built Flutter apps for different platforms at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/flutter

### VAD from file

VAD model: [Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-from-file-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-file-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-from-file-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-file-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-from-file-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-file-x86_64.apk) |
| Linux (x64) | [flutter-vad-from-file-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-file-linux-x64.zip) |
| macOS | [flutter-vad-from-file-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-file-macos.zip) |
| Windows (x64) | [flutter-vad-from-file-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-file-win-x64.zip) |
| Web | [flutter-vad-from-file-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-file-web.zip) |

### VAD from microphone

VAD model: [Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-from-microphone-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-microphone-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-from-microphone-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-microphone-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-from-microphone-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-microphone-x86_64.apk) |
| Linux (x64) | [flutter-vad-from-microphone-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-microphone-linux-x64.zip) |
| macOS | [flutter-vad-from-microphone-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-microphone-macos.zip) |
| Windows (x64) | [flutter-vad-from-microphone-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-microphone-win-x64.zip) |
| Web | [flutter-vad-from-microphone-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-from-microphone-web.zip) |

### VAD + ASR from file

VAD model: [Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)

#### Zipformer CTC (Chinese)

ASR model: [sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-zipformer-ctc-zh-web.zip) |

#### SenseVoice (Chinese/English/Japanese/Korean/Cantonese)

ASR model: [sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17](https://k2-fsa.github.io/sherpa/onnx/sense-voice/) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-sense-voice-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-sense-voice-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-sense-voice-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-sense-voice-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-sense-voice-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-sense-voice-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-sense-voice-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-sense-voice-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-sense-voice-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-sense-voice-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-sense-voice-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-sense-voice-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-sense-voice-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-sense-voice-web.zip) |

#### Whisper tiny.en (English)

ASR model: [sherpa-onnx-whisper-tiny.en](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-whisper-tiny-en-web.zip) |

#### NeMo Parakeet TDT v2 (English)

ASR model: [sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#sherpa-onnx-nemo-parakeet-tdt-0-6b-v2-int8-english) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v2-en-web.zip) |

#### NeMo Parakeet TDT v3 (25 European languages)

ASR model: [sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#sherpa-onnx-nemo-parakeet-tdt-0-6b-v3-int8-25-european-languages) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-nemo-parakeet-v3-eu-web.zip) |

#### Moonshine tiny (English)

ASR model: [sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27](https://k2-fsa.github.io/sherpa/onnx/moonshine/models-v2.html#sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27-english) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-moonshine-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-moonshine-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-moonshine-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-moonshine-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-moonshine-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-moonshine-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-moonshine-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-moonshine-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-moonshine-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-moonshine-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-moonshine-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-moonshine-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-moonshine-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-moonshine-en-web.zip) |

#### Qwen3 ASR 0.6B (multilingual)

ASR model: [sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html#sherpa-onnx-qwen3-asr-0-6b-int8-2026-03-25) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-qwen3-asr-multi-web.zip) |

#### FunASR Nano (Chinese/English/Japanese)

ASR model: [sherpa-onnx-funasr-nano-int8-2025-12-30](https://k2-fsa.github.io/sherpa/onnx/funasr-nano/pretrained.html#sherpa-onnx-funasr-nano-int8-2025-12-30-chinese-english-japanese) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-funasr-nano-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-funasr-nano-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-funasr-nano-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-funasr-nano-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-funasr-nano-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-funasr-nano-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-funasr-nano-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-funasr-nano-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-funasr-nano-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-funasr-nano-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-funasr-nano-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-funasr-nano-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-funasr-nano-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-funasr-nano-web.zip) |

#### FireRed ASR CTC v2 (Chinese/English)

ASR model: [sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25](https://k2-fsa.github.io/sherpa/onnx/FireRedAsr/pretrained.html#sherpa-onnx-fire-red-asr2-ctc-zh-en-int8-2026-02-25-v2-ctc-chinese-english-20) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-file-firered-ctc-zh-en-web.zip) |

### VAD + ASR from microphone

VAD model: [Silero VAD](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)

#### Zipformer CTC (Chinese)

ASR model: [sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-zipformer-ctc-zh-web.zip) |

#### SenseVoice (Chinese/English/Japanese/Korean/Cantonese)

ASR model: [sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17](https://k2-fsa.github.io/sherpa/onnx/sense-voice/) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-sense-voice-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-sense-voice-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-sense-voice-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-sense-voice-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-sense-voice-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-sense-voice-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-sense-voice-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-sense-voice-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-sense-voice-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-sense-voice-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-sense-voice-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-sense-voice-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-sense-voice-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-sense-voice-web.zip) |

#### Whisper tiny.en (English)

ASR model: [sherpa-onnx-whisper-tiny.en](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-whisper-tiny-en-web.zip) |

#### NeMo Parakeet TDT v2 (English)

ASR model: [sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#sherpa-onnx-nemo-parakeet-tdt-0-6b-v2-int8-english) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v2-en-web.zip) |

#### NeMo Parakeet TDT v3 (25 European languages)

ASR model: [sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/nemo-transducer-models.html#sherpa-onnx-nemo-parakeet-tdt-0-6b-v3-int8-25-european-languages) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-nemo-parakeet-v3-eu-web.zip) |

#### Moonshine tiny (English)

ASR model: [sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27](https://k2-fsa.github.io/sherpa/onnx/moonshine/models-v2.html#sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27-english) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-moonshine-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-moonshine-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-moonshine-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-moonshine-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-moonshine-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-moonshine-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-moonshine-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-moonshine-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-moonshine-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-moonshine-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-moonshine-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-moonshine-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-moonshine-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-moonshine-en-web.zip) |

#### Qwen3 ASR 0.6B (multilingual)

ASR model: [sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25](https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html#sherpa-onnx-qwen3-asr-0-6b-int8-2026-03-25) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-qwen3-asr-multi-web.zip) |

#### FunASR Nano (Chinese/English/Japanese)

ASR model: [sherpa-onnx-funasr-nano-int8-2025-12-30](https://k2-fsa.github.io/sherpa/onnx/funasr-nano/pretrained.html#sherpa-onnx-funasr-nano-int8-2025-12-30-chinese-english-japanese) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-funasr-nano-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-funasr-nano-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-funasr-nano-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-funasr-nano-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-funasr-nano-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-funasr-nano-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-funasr-nano-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-funasr-nano-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-funasr-nano-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-funasr-nano-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-funasr-nano-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-funasr-nano-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-funasr-nano-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-funasr-nano-web.zip) |

#### FireRed ASR CTC v2 (Chinese/English)

ASR model: [sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25](https://k2-fsa.github.io/sherpa/onnx/FireRedAsr/pretrained.html#sherpa-onnx-fire-red-asr2-ctc-zh-en-int8-2026-02-25-v2-ctc-chinese-english-20) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-x86_64.apk) |
| Linux (x64) | [flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-linux-x64.zip) |
| macOS | [flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-macos.zip) |
| Windows (x64) | [flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-win-x64.zip) |
| Web | [flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-vad-non-streaming-asr-from-microphone-firered-ctc-zh-en-web.zip) |

### Streaming speech recognition

#### Streaming Zipformer Bilingual (Chinese/English)

ASR model: [sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-linux-x64.zip) |
| macOS | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-win-x64.zip) |

#### Streaming Zipformer (English)

ASR model: [sherpa-onnx-streaming-zipformer-en-2023-06-26](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-26-english) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-linux-x64.zip) |
| macOS | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-en-2023-06-26-win-x64.zip) |

#### Streaming Zipformer WenetSpeech (Chinese)

ASR model: [icefall-asr-zipformer-streaming-wenetspeech-20230615](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#pkufool-icefall-asr-zipformer-streaming-wenetspeech-20230615-chinese) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/icefall-asr-zipformer-streaming-wenetspeech-20230615.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-linux-x64.zip) |
| macOS | [flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-icefall-asr-zipformer-streaming-wenetspeech-20230615-win-x64.zip) |

#### Streaming Zipformer (French)

ASR model: [sherpa-onnx-streaming-zipformer-fr-2023-04-14](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#shaojieli-sherpa-onnx-streaming-zipformer-fr-2023-04-14-french) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-fr-2023-04-14.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-linux-x64.zip) |
| macOS | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-streaming-zipformer-fr-2023-04-14-win-x64.zip) |

#### X-ASR 960ms (Chinese/English, with punctuation)

ASR model: sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05 — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-linux-x64.zip) |
| macOS | [flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-960ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-win-x64.zip) |

#### Nemotron 3.5 ASR Streaming 560ms (multilingual)

ASR model: sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11 — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-linux-x64.zip) |
| macOS | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11-win-x64.zip) |

#### X-ASR 480ms (Chinese/English, with punctuation)

ASR model: sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05 — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-linux-x64.zip) |
| macOS | [flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05-win-x64.zip) |

#### Nemotron 3.5 ASR Streaming 1120ms (multilingual)

ASR model: sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11 — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-x86_64.apk) |
| Linux (x64) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-linux-x64.zip) |
| macOS | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-macos.zip) |
| Windows (x64) | [flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-streaming_asr-sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11-win-x64.zip) |

### Text to speech

#### VITS Piper en_US-amy-low (English)

TTS model: [vits-piper-en_US-amy-low](https://k2-fsa.github.io/sherpa/onnx/tts/all/English/vits-piper-en_US-amy-low.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-vits-piper-en_US-amy-low-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-amy-low-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-vits-piper-en_US-amy-low-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-amy-low-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-vits-piper-en_US-amy-low-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-amy-low-x86_64.apk) |
| Linux (x64) | [flutter-tts-vits-piper-en_US-amy-low-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-amy-low-linux-x64.zip) |
| macOS | [flutter-tts-vits-piper-en_US-amy-low-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-amy-low-macos.zip) |
| Windows (x64) | [flutter-tts-vits-piper-en_US-amy-low-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-amy-low-win-x64.zip) |
| Web | [flutter-tts-vits-piper-en_US-amy-low-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-amy-low-web.zip) |

#### VITS Piper zh_CN-xiao_ya-medium (Chinese)

TTS model: [vits-piper-zh_CN-xiao_ya-medium](https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese/vits-piper-zh_CN-xiao_ya-medium.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-zh_CN-xiao_ya-medium.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-vits-piper-zh_CN-xiao_ya-medium-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-zh_CN-xiao_ya-medium-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-vits-piper-zh_CN-xiao_ya-medium-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-zh_CN-xiao_ya-medium-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-vits-piper-zh_CN-xiao_ya-medium-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-zh_CN-xiao_ya-medium-x86_64.apk) |
| Linux (x64) | [flutter-tts-vits-piper-zh_CN-xiao_ya-medium-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-zh_CN-xiao_ya-medium-linux-x64.zip) |
| macOS | [flutter-tts-vits-piper-zh_CN-xiao_ya-medium-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-zh_CN-xiao_ya-medium-macos.zip) |
| Windows (x64) | [flutter-tts-vits-piper-zh_CN-xiao_ya-medium-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-zh_CN-xiao_ya-medium-win-x64.zip) |
| Web | [flutter-tts-vits-piper-zh_CN-xiao_ya-medium-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-zh_CN-xiao_ya-medium-web.zip) |

#### VITS Piper en_US-libritts_r-medium (English)

TTS model: [vits-piper-en_US-libritts_r-medium](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-piper-en-us-libritts-r-medium-english-904-speakers) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-vits-piper-en_US-libritts_r-medium-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-libritts_r-medium-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-vits-piper-en_US-libritts_r-medium-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-libritts_r-medium-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-vits-piper-en_US-libritts_r-medium-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-libritts_r-medium-x86_64.apk) |
| Linux (x64) | [flutter-tts-vits-piper-en_US-libritts_r-medium-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-libritts_r-medium-linux-x64.zip) |
| macOS | [flutter-tts-vits-piper-en_US-libritts_r-medium-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-libritts_r-medium-macos.zip) |
| Windows (x64) | [flutter-tts-vits-piper-en_US-libritts_r-medium-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-libritts_r-medium-win-x64.zip) |
| Web | [flutter-tts-vits-piper-en_US-libritts_r-medium-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-piper-en_US-libritts_r-medium-web.zip) |

#### VITS Inflect en-nano-v2 (English)

TTS model: [vits-inflect-en-nano-v2](https://k2-fsa.github.io/sherpa/onnx/tts/all/English/vits-inflect-en-nano-v2.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-inflect-en-nano-v2.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-vits-inflect-en-nano-v2-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-inflect-en-nano-v2-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-vits-inflect-en-nano-v2-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-inflect-en-nano-v2-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-vits-inflect-en-nano-v2-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-inflect-en-nano-v2-x86_64.apk) |
| Linux (x64) | [flutter-tts-vits-inflect-en-nano-v2-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-inflect-en-nano-v2-linux-x64.zip) |
| macOS | [flutter-tts-vits-inflect-en-nano-v2-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-inflect-en-nano-v2-macos.zip) |
| Windows (x64) | [flutter-tts-vits-inflect-en-nano-v2-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-inflect-en-nano-v2-win-x64.zip) |
| Web | [flutter-tts-vits-inflect-en-nano-v2-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-vits-inflect-en-nano-v2-web.zip) |

#### Kokoro v0.19 (English)

TTS model: [kokoro-int8-en-v0_19](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html#kokoro-en-v0-19-english-11-speakers) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-en-v0_19.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-kokoro-int8-en-v0_19-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-int8-en-v0_19-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-kokoro-int8-en-v0_19-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-int8-en-v0_19-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-kokoro-int8-en-v0_19-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-int8-en-v0_19-x86_64.apk) |
| Linux (x64) | [flutter-tts-kokoro-int8-en-v0_19-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-int8-en-v0_19-linux-x64.zip) |
| macOS | [flutter-tts-kokoro-int8-en-v0_19-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-int8-en-v0_19-macos.zip) |
| Windows (x64) | [flutter-tts-kokoro-int8-en-v0_19-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-int8-en-v0_19-win-x64.zip) |
| Web | [flutter-tts-kokoro-int8-en-v0_19-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-int8-en-v0_19-web.zip) |

#### Kokoro v1.0 (Chinese/English)

TTS model: [kokoro-multi-lang-v1_0](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html#kokoro-multi-lang-v1-0-chinese-english-53-speakers) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-kokoro-multi-lang-v1_0-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-multi-lang-v1_0-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-kokoro-multi-lang-v1_0-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-multi-lang-v1_0-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-kokoro-multi-lang-v1_0-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-multi-lang-v1_0-x86_64.apk) |
| Linux (x64) | [flutter-tts-kokoro-multi-lang-v1_0-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-multi-lang-v1_0-linux-x64.zip) |
| macOS | [flutter-tts-kokoro-multi-lang-v1_0-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-multi-lang-v1_0-macos.zip) |
| Windows (x64) | [flutter-tts-kokoro-multi-lang-v1_0-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-multi-lang-v1_0-win-x64.zip) |
| Web | [flutter-tts-kokoro-multi-lang-v1_0-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kokoro-multi-lang-v1_0-web.zip) |

#### Matcha en_US-ljspeech (English)

TTS model: [matcha-icefall-en_US-ljspeech](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-matcha-icefall-en_US-ljspeech-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-en_US-ljspeech-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-matcha-icefall-en_US-ljspeech-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-en_US-ljspeech-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-matcha-icefall-en_US-ljspeech-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-en_US-ljspeech-x86_64.apk) |
| Linux (x64) | [flutter-tts-matcha-icefall-en_US-ljspeech-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-en_US-ljspeech-linux-x64.zip) |
| macOS | [flutter-tts-matcha-icefall-en_US-ljspeech-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-en_US-ljspeech-macos.zip) |
| Windows (x64) | [flutter-tts-matcha-icefall-en_US-ljspeech-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-en_US-ljspeech-win-x64.zip) |
| Web | [flutter-tts-matcha-icefall-en_US-ljspeech-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-en_US-ljspeech-web.zip) |

#### Matcha zh-en (Chinese/English)

TTS model: [matcha-icefall-zh-en](https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/matcha-icefall-zh-en.html) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-matcha-icefall-zh-en-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-zh-en-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-matcha-icefall-zh-en-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-zh-en-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-matcha-icefall-zh-en-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-zh-en-x86_64.apk) |
| Linux (x64) | [flutter-tts-matcha-icefall-zh-en-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-zh-en-linux-x64.zip) |
| macOS | [flutter-tts-matcha-icefall-zh-en-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-zh-en-macos.zip) |
| Windows (x64) | [flutter-tts-matcha-icefall-zh-en-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-zh-en-win-x64.zip) |
| Web | [flutter-tts-matcha-icefall-zh-en-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-matcha-icefall-zh-en-web.zip) |

#### Kitten Nano v0.1 (English)

TTS model: [kitten-nano-en-v0_1-fp16](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html#kitten-nano-en-v0-1-fp16) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-kitten-nano-en-v0_1-fp16-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kitten-nano-en-v0_1-fp16-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-kitten-nano-en-v0_1-fp16-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kitten-nano-en-v0_1-fp16-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-kitten-nano-en-v0_1-fp16-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kitten-nano-en-v0_1-fp16-x86_64.apk) |
| Linux (x64) | [flutter-tts-kitten-nano-en-v0_1-fp16-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kitten-nano-en-v0_1-fp16-linux-x64.zip) |
| macOS | [flutter-tts-kitten-nano-en-v0_1-fp16-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kitten-nano-en-v0_1-fp16-macos.zip) |
| Windows (x64) | [flutter-tts-kitten-nano-en-v0_1-fp16-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kitten-nano-en-v0_1-fp16-win-x64.zip) |
| Web | [flutter-tts-kitten-nano-en-v0_1-fp16-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-kitten-nano-en-v0_1-fp16-web.zip) |

#### PocketTTS (multilingual)

TTS model: [sherpa-onnx-pocket-tts-int8-2026-01-26](https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html#download-a-pre-trained-model) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-x86_64.apk) |
| Linux (x64) | [flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-linux-x64.zip) |
| macOS | [flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-macos.zip) |
| Windows (x64) | [flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-win-x64.zip) |
| Web | [flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-pocket-tts-int8-2026-01-26-web.zip) |

#### Supertonic 3 (multilingual)

TTS model: [sherpa-onnx-supertonic-3-tts-int8-2026-05-11](https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html#download-a-pre-trained-model) — [download](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2)

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-x86_64.apk) |
| Linux (x64) | [flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-linux-x64.zip) |
| macOS | [flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-macos.zip) |
| Windows (x64) | [flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-win-x64.zip) |
| Web | [flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-tts-sherpa-onnx-supertonic-3-tts-int8-2026-05-11-web.zip) |

### Offline punctuation restoration

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-offline-punctuation-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-offline-punctuation-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-offline-punctuation-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-offline-punctuation-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-offline-punctuation-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-offline-punctuation-x86_64.apk) |
| Linux (x64) | [flutter-offline-punctuation-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-offline-punctuation-linux-x64.zip) |
| macOS | [flutter-offline-punctuation-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-offline-punctuation-macos.zip) |
| Windows (x64) | [flutter-offline-punctuation-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-offline-punctuation-win-x64.zip) |
| Web | [flutter-offline-punctuation-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-offline-punctuation-web.zip) |

### Online punctuation restoration

| Platform | File |
|---|---|
| Android (arm64-v8a) | [flutter-online-punctuation-arm64-v8a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-online-punctuation-arm64-v8a.apk) |
| Android (armeabi-v7a) | [flutter-online-punctuation-armeabi-v7a.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-online-punctuation-armeabi-v7a.apk) |
| Android (x86_64) | [flutter-online-punctuation-x86_64.apk](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-online-punctuation-x86_64.apk) |
| Linux (x64) | [flutter-online-punctuation-linux-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-online-punctuation-linux-x64.zip) |
| macOS | [flutter-online-punctuation-macos.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-online-punctuation-macos.zip) |
| Windows (x64) | [flutter-online-punctuation-win-x64.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-online-punctuation-win-x64.zip) |
| Web | [flutter-online-punctuation-web.zip](https://github.com/k2-fsa/sherpa-onnx/releases/download/flutter/flutter-online-punctuation-web.zip) |

## Web demos

Try the following demos directly in your browser:

| Demo | URL |
|---|---|
| VAD from file | https://modelscope.cn/studios/csukuangfj/wasm-vad-from-file |
| VAD from microphone | https://modelscope.cn/studios/csukuangfj/wasm-vad-from-microphone |
| Offline punctuation restoration | https://modelscope.cn/studios/csukuangfj/wasm-offline-punctuation |
| Online punctuation restoration | https://modelscope.cn/studios/csukuangfj/wasm-online-punctuation |

## Flutter source code

| Functions | URL | Supported Platforms|
|---|---|---|
|Hello world (version info)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/hello_world)| Android, iOS, Linux, macOS, Windows, **Web**|
|Streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/streaming_asr)| Android, iOS, Linux, macOS, Windows|
|Non-streaming VAD + speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/non_streaming_vad_asr)| Android, iOS, Linux, macOS, Windows|
|Text to speech| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/tts)| Android, iOS, Linux, macOS, Windows, **Web**|
|VAD from file| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/vad-from-file)| Android, iOS, Linux, macOS, Windows, **Web**|
|VAD + ASR from file| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/vad-non-streaming-asr-from-file)| Android, iOS, Linux, macOS, Windows, **Web**|
|VAD + ASR from microphone| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/vad-non-streaming-asr-from-microphone)| Android, iOS, Linux, macOS, Windows, **Web**|
|VAD from microphone| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/vad-from-microphone)| Android, iOS, Linux, macOS, Windows, **Web**|
|Offline punctuation restoration| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/offline-punctuation)| Android, iOS, Linux, macOS, Windows, **Web**|
|Online punctuation restoration| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/online-punctuation)| Android, iOS, Linux, macOS, Windows, **Web**|

## Pure dart-examples

Hint: All of the following functions can be used in Flutter, even if some of them are only provided in pure dart api examples.

| Functions | URL | Supported Platforms|
|---|---|---|
|Streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/streaming-asr)| macOS, Windows, Linux|
|Non-Streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/non-streaming-asr)| macOS, Windows, Linux|
|Text to speech| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/tts)| macOS, Windows, Linux|
|Voice activity detection (VAD)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad)| macOS, Windows, Linux|
|VAD with non-streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad-with-non-streaming-asr)| macOS, Windows, Linux|
|Speaker identification and verification| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speaker-identification)| macOS, Windows, Linux|
|Speaker diarization| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speaker-diarization)| macOS, Windows, Linux|
|Audio tagging| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/audio-tagging)| macOS, Windows, Linux|
|Keyword spotter| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/keyword-spotter)| macOS, Windows, Linux|
|Spoken language identification| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/spoken-language-identification)| macOS, Windows, Linux|
|Add punctuations| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/add-punctuations)| macOS, Windows, Linux|
|Speech enhancement (GTCRN)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speech-enhancement-gtcrn)| macOS, Windows, Linux|
|Speech enhancement (DPDFNet)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speech-enhancement-dpdfnet)| macOS, Windows, Linux|
|Streaming speech enhancement (GTCRN)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/streaming-speech-enhancement-gtcrn)| macOS, Windows, Linux|
|Streaming speech enhancement (DPDFNet)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/streaming-speech-enhancement-dpdfnet)| macOS, Windows, Linux|
