# Introduction

This folder contains C# API examples for [sherpa-onnx][sherpa-onnx].

Please refer to the documentation
https://k2-fsa.github.io/sherpa/onnx/csharp-api/index.html
for details.

- [./speech-enhancement-gtcrn](./speech-enhancement-gtcrn) It shows how to use
  the offline speech denoiser API with GTCRN models.
- [./speech-enhancement-dpdfnet](./speech-enhancement-dpdfnet) It shows how to
  use the offline speech denoiser API with DPDFNet models. Use 16 kHz DPDFNet
  models such as `dpdfnet_baseline.onnx`, `dpdfnet2.onnx`, `dpdfnet4.onnx`, or
  `dpdfnet8.onnx` for downstream ASR and `dpdfnet2_48khz_hr.onnx` for 48 kHz
  enhancement output.
- [./streaming-speech-enhancement-gtcrn](./streaming-speech-enhancement-gtcrn)
  It shows how to use the online speech denoiser API with GTCRN models.
- [./streaming-speech-enhancement-dpdfnet](./streaming-speech-enhancement-dpdfnet)
  It shows how to use the online speech denoiser API with DPDFNet models.
- [./zipvoice-tts](./zipvoice-tts) It shows how to use ZipVoice for
  Chinese/English zero-shot text-to-speech.
- [./zipvoice-tts-play](./zipvoice-tts-play) It shows how to use ZipVoice for
  Chinese/English zero-shot text-to-speech with playback.

```bash
dotnet new console -n offline-tts-play
dotnet sln ./sherpa-onnx.sln add ./offline-tts-play
```

```bash
dotnet nuget locals all --list
dotnet nuget locals all --clear
```

[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
