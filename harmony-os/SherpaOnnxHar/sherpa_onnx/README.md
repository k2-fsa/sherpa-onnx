# Introduction

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) is one of the deployment
frameworks of [Next-gen Kaldi](https://github.com/k2-fsa).

It supports speech-to-text, text-to-speech, speaker diarization, and VAD using
onnxruntime without Internet connection.

It also supports embedded systems, Android, iOS, HarmonyOS,
Raspberry Pi, RISC-V, x86_64 servers, websocket server/client,
C/C++, Python, Kotlin, C#, Go, NodeJS, Java, Swift, Dart, JavaScript,
Flutter, Object Pascal, Lazarus, Rust, etc.


# Installation

To use `sherpa-onnx` in your project, please either use

```
ohpm install sherpa_onnx
```
or update your `oh-package.json5` to include the following:

```
  "dependencies": {
    "sherpa_onnx": "1.10.33",
  },
```

Note that we recommend always using the latest version.

# Examples

| Demo | URL | Description|
|------|-----|------------|
|SherpaOnnxVadAsr|[Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/harmony-os/SherpaOnnxVadAsr)|It shows how to use VAD with a non-streaming ASR model for on-device speech recognition without accessing the network |
|SherpaOnnxTts|[Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/harmony-os/SherpaOnnxTts)|It shows how to use Next-gen Kaldi for on-device text-to-speech (TTS, i.e., speech synthesis)|

# Documentation

If you have any issues, please either look at our doc at
<https://k2-fsa.github.io/sherpa/onnx/> or create an issue at
<https://github.com/k2-fsa/sherpa-onnx/issues>
