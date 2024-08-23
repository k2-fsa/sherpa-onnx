### Supported functions

|Speech recognition| Speech synthesis | Speaker verification | Speaker identification |
|------------------|------------------|----------------------|------------------------|
|   ✔️              |         ✔️        |          ✔️           |                ✔️       |

| Spoken Language identification | Audio tagging | Voice activity detection |
|--------------------------------|---------------|--------------------------|
|                 ✔️              |          ✔️    |                ✔️         |

| Keyword spotting | Add punctuation |
|------------------|-----------------|
|     ✔️            |       ✔️         |

### Supported platforms

|Architecture| Android          | iOS           | Windows    | macOS | linux |
|------------|------------------|---------------|------------|-------|-------|
|   x64      |  ✔️               |               |   ✔️        | ✔️     |  ✔️    |
|   x86      |  ✔️               |               |   ✔️        |       |       |
|   arm64    |  ✔️               | ✔️             |   ✔️        | ✔️     |  ✔️    |
|   arm32    |  ✔️               |               |            |       |  ✔️    |
|   riscv64  |                  |               |            |       |  ✔️    |


### Supported programming languages

| 1. C++ | 2. C  | 3. Python | 4. JavaScript |
|--------|-------|-----------|---------------|
|   ✔️    | ✔️     | ✔️         |    ✔️          |

|5. Java | 6. C# | 7. Kotlin | 8. Swift |
|--------|-------|-----------|----------|
| ✔️      |  ✔️    | ✔️         |  ✔️       |

| 9. Go | 10. Dart | 11. Rust | 12. Pascal |
|-------|----------|----------|------------|
| ✔️     |  ✔️       |   ✔️      |    ✔️       |

For Rust support, please see https://github.com/thewh1teagle/sherpa-rs

It also supports WebAssembly.

## Introduction

This repository supports running the following functions **locally**

  - Speech-to-text (i.e., ASR); both streaming and non-streaming are supported
  - Text-to-speech (i.e., TTS)
  - Speaker identification
  - Speaker verification
  - Spoken language identification
  - Audio tagging
  - VAD (e.g., [silero-vad](https://github.com/snakers4/silero-vad))
  - Keyword spotting

on the following platforms and operating systems:

  - x86, ``x86_64``, 32-bit ARM, 64-bit ARM (arm64, aarch64), RISC-V (riscv64)
  - Linux, macOS, Windows, openKylin
  - Android, WearOS
  - iOS
  - NodeJS
  - WebAssembly
  - [Raspberry Pi](https://www.raspberrypi.com/)
  - [RV1126](https://www.rock-chips.com/uploads/pdf/2022.8.26/191/RV1126%20Brief%20Datasheet.pdf)
  - [LicheePi4A](https://sipeed.com/licheepi4a)
  - [VisionFive 2](https://www.starfivetech.com/en/site/boards)
  - [旭日X3派](https://developer.horizon.ai/api/v1/fileData/documents_pi/index.html)
  - etc

with the following APIs

  - C++, C, Python, Go, ``C#``
  - Java, Kotlin, JavaScript
  - Swift, Rust
  - Dart, Object Pascal

### Links for Huggingface Spaces

You can visit the following Huggingface spaces to try `sherpa-onnx` without
installing anything. All you need is a browser.

| Description | URL |
|---|---|
| Speech recognition | [Click me](https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition)|
| Speech recognition with [Whisper](https://github.com/openai/whisper)| [Click me](https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition-with-whisper)|
| Speech synthesis | [Click me](https://huggingface.co/spaces/k2-fsa/text-to-speech)|
| Generate subtitles| [Click me](https://huggingface.co/spaces/k2-fsa/generate-subtitles-for-videos)|
|Audio tagging| [Click me](https://huggingface.co/spaces/k2-fsa/audio-tagging)|
|Spoken language identification with [Whisper](https://github.com/openai/whisper)|[Click me](https://huggingface.co/spaces/k2-fsa/spoken-language-identification)|

We also have spaces built using WebAssembly. The are listed below:

| Description | URL| Chinese users|
|---|---|---|
|Voice activity detection with [silero-vad](https://github.com/snakers4/silero-vad)| [Click me](https://huggingface.co/spaces/k2-fsa/web-assembly-vad-sherpa-onnx)|[地址](https://modelscope.cn/studios/csukuangfj/web-assembly-vad-sherpa-onnx)|
|Real-time speech recognition (Chinese + English) with Zipformer | [Click me](https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en)|[地址](https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en)|
|Real-time speech recognition (Chinese + English) with Paraformer|[Click me](https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en-paraformer)| [地址](https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en-paraformer)|
|Real-time speech recognition (Chinese + English + Cantonese) with Paraformer|[Click me](https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-cantonese-en-paraformer)| [地址](https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-cantonese-en-paraformer)|
|Real-time speech recognition (English) |[Click me](https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-en)|[地址](https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-en)|
|Speech synthesis (English) |[Click me](https://huggingface.co/spaces/k2-fsa/web-assembly-tts-sherpa-onnx-en)| [地址](https://modelscope.cn/studios/k2-fsa/web-assembly-tts-sherpa-onnx-en)|
|Speech synthesis (German)|[Click me](https://huggingface.co/spaces/k2-fsa/web-assembly-tts-sherpa-onnx-de)| [地址](https://modelscope.cn/studios/k2-fsa/web-assembly-tts-sherpa-onnx-de)|

### Links for pre-built Android APKs

| Description                    | URL                                                                                     | 中国用户                                                                             |
|--------------------------------|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Streaming speech recognition             | [Address](https://k2-fsa.github.io/sherpa/onnx/android/apk.html)                        | [点此](https://k2-fsa.github.io/sherpa/onnx/android/apk-cn.html)                        |
| Text-to-speech | [Address](https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html)                     | [点此](https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine-cn.html)                     |
|Voice activity detection (VAD) | [Address](https://k2-fsa.github.io/sherpa/onnx/vad/apk.html) | [点此](https://k2-fsa.github.io/sherpa/onnx/vad/apk-cn.html)|
|VAD + non-streaming speech recognition| [Address](https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr.html)| [点此](https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr-cn.html)|
|Two-pass speech recognition| [Address](https://k2-fsa.github.io/sherpa/onnx/android/apk-2pass.html)| [点此](https://k2-fsa.github.io/sherpa/onnx/android/apk-2pass-cn.html)|
| Audio tagging                  | [Address](https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk.html)                  | [点此](https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-cn.html)                  |
| Audio tagging (WearOS)         | [Address](https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-wearos.html)           | [点此](https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-wearos-cn.html)           |
| Speaker identification         | [Address](https://k2-fsa.github.io/sherpa/onnx/speaker-identification/apk.html)         | [点此](https://k2-fsa.github.io/sherpa/onnx/speaker-identification/apk-cn.html)         |
| Spoken language identification | [Address](https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/apk.html) | [点此](https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/apk-cn.html) |
|Keyword spotting| [Address](https://k2-fsa.github.io/sherpa/onnx/kws/apk.html)| [点此](https://k2-fsa.github.io/sherpa/onnx/kws/apk-cn.html)|

### Links for pre-built Flutter APPs

#### Real-time speech recognition

| Description                    | URL                                                                 | 中国用户                                                            |
|--------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|
| Streaming speech recognition   | [Address](https://k2-fsa.github.io/sherpa/onnx/flutter/asr/app.html)| [点此](https://k2-fsa.github.io/sherpa/onnx/flutter/asr/app-cn.html)|

#### Text-to-speech

| Description                    | URL                                                          | 中国用户                                                                    |
|--------------------------------|--------------------------------------------------------------|-----------------------------------------------------------------------------|
| Android (arm64-v8a, armeabi-v7a, x86_64) | [Address](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-android.html) | [点此](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-android-cn.html)|
| Linux (x64)    | [Address](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-linux.html)       | [点此](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-linux-cn.html)      |
| macOS (x64)    | [Address](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-x64.html)   | [点此](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-x64-cn.html)  |
| macOS (arm64)  | [Address](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-arm64.html) | [点此](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-arm64-cn.html)|
| Windows (x64)  | [Address](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-win.html)         | [点此](https://k2-fsa.github.io/sherpa/onnx/flutter/tts-win-cn.html)        |

> Note: You need to build from source for iOS.

### Links for pre-built Lazarus APPs

#### Generating subtitles

| Description                    | URL                                                                 | 中国用户                                                            |
|--------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|
| Generate subtitles (生成字幕)   | [Address](https://k2-fsa.github.io/sherpa/onnx/lazarus/download-generated-subtitles.html)| [点此](https://k2-fsa.github.io/sherpa/onnx/lazarus/download-generated-subtitles-cn.html)|


### Links for pre-trained models

| Description                    | URL                                                                                                                            |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Speech recognition (speech to text, ASR)             | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)              |
| Text-to-speech (TTS)                 | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)                             |
| VAD | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)|
| Keyword spotting |[Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models)|
| Audio tagging                  | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models)|
| Speaker identification (Speaker ID)         | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)|
| Spoken language identification (Language ID) | See multi-lingual [Whisper](https://github.com/openai/whisper) ASR models from  [Speech recognition](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) |
| Punctuation| [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models)|

### Useful links

- Documentation: https://k2-fsa.github.io/sherpa/onnx/
- Bilibili 演示视频: https://search.bilibili.com/all?keyword=%E6%96%B0%E4%B8%80%E4%BB%A3Kaldi

### How to reach us

Please see
https://k2-fsa.github.io/sherpa/social-groups.html
for 新一代 Kaldi **微信交流群** and **QQ 交流群**.
