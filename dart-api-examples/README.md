# Introduction

This directory contains examples for Dart API.

You can find the package at
https://pub.dev/packages/sherpa_onnx

## Description

| Directory | Description |
|-----------|-------------|
| [./speaker-diarization](./speaker-diarization)| Example for speaker diarization.|
| [./add-punctuations](./add-punctuations)| Example for adding punctuations to text.|
| [./audio-tagging](./audio-tagging)| Example for audio tagging.|
| [./keyword-spotter](./keyword-spotter)| Example for keyword spotting|
| [./non-streaming-asr](./non-streaming-asr)| Example for non-streaming speech recognition|
| [./speaker-identification](./speaker-identification)| Example for speaker identification and verification.|
| [./streaming-asr](./streaming-asr)| Example for streaming speech recognition|
| [./tts](./tts)| Example for text to speech|
| [./vad-with-non-streaming-asr](./vad-with-non-streaming-asr)| Example for voice activity detection with non-streaming speech recognition. You can use it to generate subtitles.|
| [./vad](./vad)| Example for voice activity detection|
| [./speech-enhancement-gtcrn](./speech-enhancement-gtcrn)| Example for speech enhancement/denoising with GTCRN.|
| [./speech-enhancement-dpdfnet](./speech-enhancement-dpdfnet)| Example for speech enhancement/denoising with DPDFNet, including the 16 kHz family (`dpdfnet_baseline`, `dpdfnet2`, `dpdfnet4`, `dpdfnet8`).|
| [./streaming-speech-enhancement-gtcrn](./streaming-speech-enhancement-gtcrn)| Example for streaming speech enhancement/denoising with GTCRN.|
| [./streaming-speech-enhancement-dpdfnet](./streaming-speech-enhancement-dpdfnet)| Example for streaming speech enhancement/denoising with DPDFNet.|

## How to create an example in this folder

```bash
dart create vad
cd vad

# Edit pubspec.yaml and add sherpa_onnx to dependencies

dart pub get
dart run
```
