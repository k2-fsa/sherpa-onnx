# Introduction

This directory contains examples for how to use the [Object Pascal](https://en.wikipedia.org/wiki/Object_Pascal)
APIs of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

**Documentation for this directory**:
https://k2-fsa.github.io/sherpa/onnx/pascal-api/index.html

|Directory| Description|
|---------|------------|
|[read-wav](./read-wav)|It shows how to read a wave file.|
|[speaker-diarization](./speaker-diarization)|It shows how to use Pascal API for speaker diarization.|
|[speech-enhancement-gtcrn](./speech-enhancement-gtcrn)| It shows how to use the offline speech denoiser API with GTCRN.|
|[speech-enhancement-dpdfnet](./speech-enhancement-dpdfnet)| It shows how to use the offline speech denoiser API with DPDFNet. Use `dpdfnet_baseline.onnx`, `dpdfnet2.onnx`, `dpdfnet4.onnx`, or `dpdfnet8.onnx` for 16 kHz downstream ASR and `dpdfnet2_48khz_hr.onnx` for 48 kHz enhancement output.|
|[streaming-speech-enhancement-gtcrn](./streaming-speech-enhancement-gtcrn)| It shows how to use the streaming speech denoiser API with GTCRN.|
|[streaming-speech-enhancement-dpdfnet](./streaming-speech-enhancement-dpdfnet)| It shows how to use the streaming speech denoiser API with DPDFNet.|
|[streaming-asr](./streaming-asr)| It shows how to use streaming models for speech recognition.|
|[non-streaming-asr](./non-streaming-asr)| It shows how to use non-streaming models for speech recognition.|
|[vad](./vad)| It shows how to use the voice activity detection API.|
|[vad-with-non-streaming-asr](./vad-with-non-streaming-asr)| It shows how to use the voice activity detection API with non-streaming models for speech recognition.|
|[portaudio-test](./portaudio-test)| It shows how to use PortAudio for recording and playing.|
|[tts](./tts)| It shows how to use the text-to-speech API.|
