# Introduction

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
  - Swift

## Links for pre-built Android APKs

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

## Links for pre-trained models

| Description                    | URL                                                                                                                            |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| Speech recognition (speech to text, ASR)             | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)              |
| Text-to-speech (TTS)                 | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)                             |
| VAD | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx)|
| Keyword spotting |[Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models)|
| Audio tagging                  | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models)|
| Speaker identification (Speaker ID)         | [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)|
| Spoken language identification (Language ID) | See multi-lingual Whisper ASR models from  [Speech recognition](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) |
| Punctuation| [Address](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models)|

## Useful links

- Documentation: https://k2-fsa.github.io/sherpa/onnx/
- Bilibili 演示视频: https://search.bilibili.com/all?keyword=%E6%96%B0%E4%B8%80%E4%BB%A3Kaldi

## How to reach us

Please see
https://k2-fsa.github.io/sherpa/social-groups.html
for 新一代 Kaldi **微信交流群** and **QQ 交流群**.
