### Supported functions

|Speech recognition| Speech synthesis |
|------------------|------------------|
|   ✔️              |         ✔️        |

|Speaker identification| Speaker diarization | Speaker verification |
|----------------------|-------------------- |------------------------|
|   ✔️                  |         ✔️           |            ✔️           |

| Spoken Language identification | Audio tagging | Voice activity detection |
|--------------------------------|---------------|--------------------------|
|                 ✔️              |          ✔️    |                ✔️         |

| Keyword spotting | Add punctuation |
|------------------|-----------------|
|     ✔️            |       ✔️         |

### Supported platforms

|Architecture| Android | iOS     | Windows    | macOS | linux |
|------------|---------|---------|------------|-------|-------|
|   x64      |  ✔️      |         |   ✔️        | ✔️     |  ✔️    |
|   x86      |  ✔️      |         |   ✔️        |       |       |
|   arm64    |  ✔️      | ✔️       |   ✔️        | ✔️     |  ✔️    |
|   arm32    |  ✔️      |         |            |       |  ✔️    |
|   riscv64  |         |         |            |       |  ✔️    |


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

For Rust support, please see [sherpa-rs][sherpa-rs]

It also supports WebAssembly.

## Introduction

This repository supports running the following functions **locally**

  - Speech-to-text (i.e., ASR); both streaming and non-streaming are supported
  - Text-to-speech (i.e., TTS)
  - Speaker diarization
  - Speaker identification
  - Speaker verification
  - Spoken language identification
  - Audio tagging
  - VAD (e.g., [silero-vad][silero-vad])
  - Keyword spotting

on the following platforms and operating systems:

  - x86, ``x86_64``, 32-bit ARM, 64-bit ARM (arm64, aarch64), RISC-V (riscv64)
  - Linux, macOS, Windows, openKylin
  - Android, WearOS
  - iOS
  - NodeJS
  - WebAssembly
  - [Raspberry Pi][Raspberry Pi]
  - [RV1126][RV1126]
  - [LicheePi4A][LicheePi4A]
  - [VisionFive 2][VisionFive 2]
  - [旭日X3派][旭日X3派]
  - [爱芯派][爱芯派]
  - etc

with the following APIs

  - C++, C, Python, Go, ``C#``
  - Java, Kotlin, JavaScript
  - Swift, Rust
  - Dart, Object Pascal

### Links for Huggingface Spaces

<details>
<summary>You can visit the following Huggingface spaces to try sherpa-onnx without
installing anything. All you need is a browser.</summary>

| Description                                           | URL                                     |
|-------------------------------------------------------|-----------------------------------------|
| Speaker diarization                                   | [Click me][hf-space-speaker-diarization]|
| Speech recognition                                    | [Click me][hf-space-asr]                |
| Speech recognition with [Whisper][Whisper]            | [Click me][hf-space-asr-whisper]        |
| Speech synthesis                                      | [Click me][hf-space-tts]                |
| Generate subtitles                                    | [Click me][hf-space-subtitle]           |
| Audio tagging                                         | [Click me][hf-space-audio-tagging]      |
| Spoken language identification with [Whisper][Whisper]| [Click me][hf-space-slid-whisper]       |

We also have spaces built using WebAssembly. They are listed below:

| Description                                                                              | Huggingface space| ModelScope space|
|------------------------------------------------------------------------------------------|------------------|-----------------|
|Voice activity detection with [silero-vad][silero-vad]                                    | [Click me][wasm-hf-vad]|[地址][wasm-ms-vad]|
|Real-time speech recognition (Chinese + English) with Zipformer                           | [Click me][wasm-hf-streaming-asr-zh-en-zipformer]|[地址][wasm-hf-streaming-asr-zh-en-zipformer]|
|Real-time speech recognition (Chinese + English) with Paraformer                          |[Click me][wasm-hf-streaming-asr-zh-en-paraformer]| [地址][wasm-ms-streaming-asr-zh-en-paraformer]|
|Real-time speech recognition (Chinese + English + Cantonese) with [Paraformer-large][Paraformer-large]|[Click me][wasm-hf-streaming-asr-zh-en-yue-paraformer]| [地址][wasm-ms-streaming-asr-zh-en-yue-paraformer]|
|Real-time speech recognition (English) |[Click me][wasm-hf-streaming-asr-en-zipformer]    |[地址][wasm-ms-streaming-asr-en-zipformer]|
|VAD + speech recognition (Chinese + English + Korean + Japanese + Cantonese) with [SenseVoice][SenseVoice]|[Click me][wasm-hf-vad-asr-zh-en-ko-ja-yue-sense-voice]| [地址][wasm-ms-vad-asr-zh-en-ko-ja-yue-sense-voice]|
|VAD + speech recognition (English) with [Whisper][Whisper] tiny.en|[Click me][wasm-hf-vad-asr-en-whisper-tiny-en]| [地址][wasm-ms-vad-asr-en-whisper-tiny-en]|
|VAD + speech recognition (English) with Zipformer trained with [GigaSpeech][GigaSpeech]    |[Click me][wasm-hf-vad-asr-en-zipformer-gigaspeech]| [地址][wasm-ms-vad-asr-en-zipformer-gigaspeech]|
|VAD + speech recognition (Chinese) with Zipformer trained with [WenetSpeech][WenetSpeech]  |[Click me][wasm-hf-vad-asr-zh-zipformer-wenetspeech]| [地址][wasm-ms-vad-asr-zh-zipformer-wenetspeech]|
|VAD + speech recognition (Japanese) with Zipformer trained with [ReazonSpeech][ReazonSpeech]|[Click me][wasm-hf-vad-asr-ja-zipformer-reazonspeech]| [地址][wasm-ms-vad-asr-ja-zipformer-reazonspeech]|
|VAD + speech recognition (Thai) with Zipformer trained with [GigaSpeech2][GigaSpeech2]      |[Click me][wasm-hf-vad-asr-th-zipformer-gigaspeech2]| [地址][wasm-ms-vad-asr-th-zipformer-gigaspeech2]|
|VAD + speech recognition (Chinese 多种方言) with a [TeleSpeech-ASR][TeleSpeech-ASR] CTC model|[Click me][wasm-hf-vad-asr-zh-telespeech]| [地址][wasm-ms-vad-asr-zh-telespeech]|
|VAD + speech recognition (English + Chinese, 及多种中文方言) with Paraformer-large          |[Click me][wasm-hf-vad-asr-zh-en-paraformer-large]| [地址][wasm-ms-vad-asr-zh-en-paraformer-large]|
|VAD + speech recognition (English + Chinese, 及多种中文方言) with Paraformer-small          |[Click me][wasm-hf-vad-asr-zh-en-paraformer-small]| [地址][wasm-ms-vad-asr-zh-en-paraformer-small]|
|Speech synthesis (English)                                                                  |[Click me][wasm-hf-tts-piper-en]| [地址][wasm-ms-tts-piper-en]|
|Speech synthesis (German)                                                                   |[Click me][wasm-hf-tts-piper-de]| [地址][wasm-ms-tts-piper-de]|
|Speaker diarization                                                                         |[Click me][wasm-hf-speaker-diarization]|[地址][wasm-ms-speaker-diarization]|

</details>

### Links for pre-built Android APKs

<details>

<summary>You can find pre-built Android APKs for this repository in the following table</summary>

| Description                            | URL                                | 中国用户                          |
|----------------------------------------|------------------------------------|-----------------------------------|
| Speaker diarization                    | [Address][apk-speaker-diarization] | [点此][apk-speaker-diarization-cn]|
| Streaming speech recognition           | [Address][apk-streaming-asr]       | [点此][apk-streaming-asr-cn]      |
| Text-to-speech                         | [Address][apk-tts]                 | [点此][apk-tts-cn]                |
| Voice activity detection (VAD)         | [Address][apk-vad]                 | [点此][apk-vad-cn]                |
| VAD + non-streaming speech recognition | [Address][apk-vad-asr]             | [点此][apk-vad-asr-cn]            |
| Two-pass speech recognition            | [Address][apk-2pass]               | [点此][apk-2pass-cn]              |
| Audio tagging                          | [Address][apk-at]                  | [点此][apk-at-cn]                 |
| Audio tagging (WearOS)                 | [Address][apk-at-wearos]           | [点此][apk-at-wearos-cn]          |
| Speaker identification                 | [Address][apk-sid]                 | [点此][apk-sid-cn]                |
| Spoken language identification         | [Address][apk-slid]                | [点此][apk-slid-cn]               |
| Keyword spotting                       | [Address][apk-kws]                 | [点此][apk-kws-cn]                |

</details>

### Links for pre-built Flutter APPs

<details>

#### Real-time speech recognition

| Description                    | URL                                 | 中国用户                            |
|--------------------------------|-------------------------------------|-------------------------------------|
| Streaming speech recognition   | [Address][apk-flutter-streaming-asr]| [点此][apk-flutter-streaming-asr-cn]|

#### Text-to-speech

| Description                              | URL                                | 中国用户                           |
|------------------------------------------|------------------------------------|------------------------------------|
| Android (arm64-v8a, armeabi-v7a, x86_64) | [Address][flutter-tts-android]     | [点此][flutter-tts-android-cn]     |
| Linux (x64)                              | [Address][flutter-tts-linux]       | [点此][flutter-tts-linux-cn]       |
| macOS (x64)                              | [Address][flutter-tts-macos-x64]   | [点此][flutter-tts-macos-arm64-cn] |
| macOS (arm64)                            | [Address][flutter-tts-macos-arm64] | [点此][flutter-tts-macos-x64-cn]   |
| Windows (x64)                            | [Address][flutter-tts-win-x64]     | [点此][flutter-tts-win-x64-cn]     |

> Note: You need to build from source for iOS.

</details>

### Links for pre-built Lazarus APPs

<details>

#### Generating subtitles

| Description                    | URL                        | 中国用户                   |
|--------------------------------|----------------------------|----------------------------|
| Generate subtitles (生成字幕)  | [Address][lazarus-subtitle]| [点此][lazarus-subtitle-cn]|

</details>

### Links for pre-trained models

<details>

| Description                                 | URL                                                                                   |
|---------------------------------------------|---------------------------------------------------------------------------------------|
| Speech recognition (speech to text, ASR)    | [Address][asr-models]                                                                 |
| Text-to-speech (TTS)                        | [Address][tts-models]                                                                 |
| VAD                                         | [Address][vad-models]                                                                 |
| Keyword spotting                            | [Address][kws-models]                                                                 |
| Audio tagging                               | [Address][at-models]                                                                  |
| Speaker identification (Speaker ID)         | [Address][sid-models]                                                                 |
| Spoken language identification (Language ID)| See multi-lingual [Whisper][Whisper] ASR models from  [Speech recognition][asr-models]|
| Punctuation                                 | [Address][punct-models]                                                               |
| Speaker segmentation                        | [Address][speaker-segmentation-models]                                                |

</details>

### Useful links

- Documentation: https://k2-fsa.github.io/sherpa/onnx/
- Bilibili 演示视频: https://search.bilibili.com/all?keyword=%E6%96%B0%E4%B8%80%E4%BB%A3Kaldi

### How to reach us

Please see
https://k2-fsa.github.io/sherpa/social-groups.html
for 新一代 Kaldi **微信交流群** and **QQ 交流群**.

## Projects using sherpa-onnx

### [voiceapi](https://github.com/ruzhila/voiceapi)

<details>
  <summary>Streaming ASR and TTS based on FastAPI</summary>


It shows how to use the ASR and TTS Python APIs with FastAPI.
</details>

### [腾讯会议摸鱼工具 TMSpeech](https://github.com/jxlpzqc/TMSpeech)

Uses streaming ASR in C# with graphical user interface.

Video demo in Chinese: [【开源】Windows实时字幕软件（网课/开会必备）](https://www.bilibili.com/video/BV1rX4y1p7Nx)

### [lol互动助手](https://github.com/l1veIn/lol-wom-electron)

It uses the JavaScript API of sherpa-onnx along with [Electron](https://electronjs.org/)

Video demo in Chinese: [爆了！炫神教你开打字挂！真正影响胜率的英雄联盟工具！英雄联盟的最后一块拼图！和游戏中的每个人无障碍沟通！](https://www.bilibili.com/video/BV142tje9E74)


[sherpa-rs]: https://github.com/thewh1teagle/sherpa-rs
[silero-vad]: https://github.com/snakers4/silero-vad
[Raspberry Pi]: https://www.raspberrypi.com/
[RV1126]: https://www.rock-chips.com/uploads/pdf/2022.8.26/191/RV1126%20Brief%20Datasheet.pdf
[LicheePi4A]: https://sipeed.com/licheepi4a
[VisionFive 2]: https://www.starfivetech.com/en/site/boards
[旭日X3派]: https://developer.horizon.ai/api/v1/fileData/documents_pi/index.html
[爱芯派]: https://wiki.sipeed.com/hardware/zh/maixIII/ax-pi/axpi.html
[hf-space-speaker-diarization]: https://huggingface.co/spaces/k2-fsa/speaker-diarization
[hf-space-asr]: https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition
[Whisper]: https://github.com/openai/whisper
[hf-space-asr-whisper]: https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition-with-whisper
[hf-space-tts]: https://huggingface.co/spaces/k2-fsa/text-to-speech
[hf-space-subtitle]: https://huggingface.co/spaces/k2-fsa/generate-subtitles-for-videos
[hf-space-audio-tagging]: https://huggingface.co/spaces/k2-fsa/audio-tagging
[hf-space-slid-whisper]: https://huggingface.co/spaces/k2-fsa/spoken-language-identification
[wasm-hf-vad]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-sherpa-onnx
[wasm-ms-vad]: https://modelscope.cn/studios/csukuangfj/web-assembly-vad-sherpa-onnx
[wasm-hf-streaming-asr-zh-en-zipformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en
[wasm-ms-streaming-asr-zh-en-zipformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en
[wasm-hf-streaming-asr-zh-en-paraformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en-paraformer
[wasm-ms-streaming-asr-zh-en-paraformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-en-paraformer
[Paraformer-large]: https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary
[wasm-hf-streaming-asr-zh-en-yue-paraformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-zh-cantonese-en-paraformer
[wasm-ms-streaming-asr-zh-en-yue-paraformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-zh-cantonese-en-paraformer
[wasm-hf-streaming-asr-en-zipformer]: https://huggingface.co/spaces/k2-fsa/web-assembly-asr-sherpa-onnx-en
[wasm-ms-streaming-asr-en-zipformer]: https://modelscope.cn/studios/k2-fsa/web-assembly-asr-sherpa-onnx-en
[SenseVoice]: https://github.com/FunAudioLLM/SenseVoice
[wasm-hf-vad-asr-zh-en-ko-ja-yue-sense-voice]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-ja-ko-cantonese-sense-voice
[wasm-ms-vad-asr-zh-en-ko-ja-yue-sense-voice]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-zh-en-jp-ko-cantonese-sense-voice
[wasm-hf-vad-asr-en-whisper-tiny-en]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-whisper-tiny
[wasm-ms-vad-asr-en-whisper-tiny-en]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-en-whisper-tiny
[wasm-hf-vad-asr-en-zipformer-gigaspeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-zipformer-gigaspeech
[wasm-ms-vad-asr-en-zipformer-gigaspeech]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-en-zipformer-gigaspeech
[wasm-hf-vad-asr-zh-zipformer-wenetspeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-wenetspeech
[wasm-ms-vad-asr-zh-zipformer-wenetspeech]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-zipformer-wenetspeech
[ReazonSpeech]: https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf
[wasm-hf-vad-asr-ja-zipformer-reazonspeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-ja-zipformer
[wasm-ms-vad-asr-ja-zipformer-reazonspeech]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-ja-zipformer
[GigaSpeech2]: https://github.com/SpeechColab/GigaSpeech2
[wasm-hf-vad-asr-th-zipformer-gigaspeech2]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-th-zipformer
[wasm-ms-vad-asr-th-zipformer-gigaspeech2]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-vad-asr-sherpa-onnx-th-zipformer
[TeleSpeech-ASR]: https://github.com/Tele-AI/TeleSpeech-ASR
[wasm-hf-vad-asr-zh-telespeech]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-telespeech
[wasm-ms-vad-asr-zh-telespeech]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-telespeech
[wasm-hf-vad-asr-zh-en-paraformer-large]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer
[wasm-ms-vad-asr-zh-en-paraformer-large]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer
[wasm-hf-vad-asr-zh-en-paraformer-small]: https://huggingface.co/spaces/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer-small
[wasm-ms-vad-asr-zh-en-paraformer-small]: https://www.modelscope.cn/studios/k2-fsa/web-assembly-vad-asr-sherpa-onnx-zh-en-paraformer-small
[wasm-hf-tts-piper-en]: https://huggingface.co/spaces/k2-fsa/web-assembly-tts-sherpa-onnx-en
[wasm-ms-tts-piper-en]: https://modelscope.cn/studios/k2-fsa/web-assembly-tts-sherpa-onnx-en
[wasm-hf-tts-piper-de]: https://huggingface.co/spaces/k2-fsa/web-assembly-tts-sherpa-onnx-de
[wasm-ms-tts-piper-de]: https://modelscope.cn/studios/k2-fsa/web-assembly-tts-sherpa-onnx-de
[wasm-hf-speaker-diarization]: https://huggingface.co/spaces/k2-fsa/web-assembly-speaker-diarization-sherpa-onnx
[wasm-ms-speaker-diarization]: https://www.modelscope.cn/studios/csukuangfj/web-assembly-speaker-diarization-sherpa-onnx
[apk-speaker-diarization]: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/apk.html
[apk-speaker-diarization-cn]: https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/apk-cn.html
[apk-streaming-asr]: https://k2-fsa.github.io/sherpa/onnx/android/apk.html
[apk-streaming-asr-cn]: https://k2-fsa.github.io/sherpa/onnx/android/apk-cn.html
[apk-tts]: https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html
[apk-tts-cn]: https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine-cn.html
[apk-vad]: https://k2-fsa.github.io/sherpa/onnx/vad/apk.html
[apk-vad-cn]: https://k2-fsa.github.io/sherpa/onnx/vad/apk-cn.html
[apk-vad-asr]: https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr.html
[apk-vad-asr-cn]: https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr-cn.html
[apk-2pass]: https://k2-fsa.github.io/sherpa/onnx/android/apk-2pass.html
[apk-2pass-cn]: https://k2-fsa.github.io/sherpa/onnx/android/apk-2pass-cn.html
[apk-at]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk.html
[apk-at-cn]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-cn.html
[apk-at-wearos]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-wearos.html
[apk-at-wearos-cn]: https://k2-fsa.github.io/sherpa/onnx/audio-tagging/apk-wearos-cn.html
[apk-sid]: https://k2-fsa.github.io/sherpa/onnx/speaker-identification/apk.html
[apk-sid-cn]: https://k2-fsa.github.io/sherpa/onnx/speaker-identification/apk-cn.html
[apk-slid]: https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/apk.html
[apk-slid-cn]: https://k2-fsa.github.io/sherpa/onnx/spoken-language-identification/apk-cn.html
[apk-kws]: https://k2-fsa.github.io/sherpa/onnx/kws/apk.html
[apk-kws-cn]: https://k2-fsa.github.io/sherpa/onnx/kws/apk-cn.html
[apk-flutter-streaming-asr]: https://k2-fsa.github.io/sherpa/onnx/flutter/asr/app.html
[apk-flutter-streaming-asr-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/asr/app-cn.html
[flutter-tts-android]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-android.html
[flutter-tts-android-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-android-cn.html
[flutter-tts-linux]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-linux.html
[flutter-tts-linux-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-linux-cn.html
[flutter-tts-macos-x64]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-x64.html
[flutter-tts-macos-arm64-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-x64-cn.html
[flutter-tts-macos-arm64]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-arm64.html
[flutter-tts-macos-x64-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-macos-arm64-cn.html
[flutter-tts-win-x64]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-win.html
[flutter-tts-win-x64-cn]: https://k2-fsa.github.io/sherpa/onnx/flutter/tts-win-cn.html
[lazarus-subtitle]: https://k2-fsa.github.io/sherpa/onnx/lazarus/download-generated-subtitles.html
[lazarus-subtitle-cn]: https://k2-fsa.github.io/sherpa/onnx/lazarus/download-generated-subtitles-cn.html
[asr-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
[tts-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
[vad-models]: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
[kws-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
[at-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
[sid-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
[slid-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
[punct-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
[speaker-segmentation-models]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
[GigaSpeech]: https://github.com/SpeechColab/GigaSpeech
[WenetSpeech]: https://github.com/wenet-e2e/WenetSpeech
