# Introduction

This directory contains examples for Dart API.

You can find the package at
https://pub.dev/packages/sherpa_onnx

## Descirption

| Directory | Description |
|-----------|-------------|
| [./keyword-spotter](./keyword-spotter)| Example for keyword spotting|
| [./non-streaming-asr](./non-streaming-asr)| Example for non-streaming speech recognition|
| [./streaming-asr](./streaming-asr)| Example for streaming speech recognition|
| [./tts](./tts)| Example for text to speech|
| [./vad](./vad)| Example for voice activity detection|
| [./vad-with-non-streaming-asr](./vad-with-non-streaming-asr)| Example for voice activity detection with non-streaming speech recognition. You can use it to generate subtitles.|

## How to create an example in this folder

```bash
dart create vad
cd vad

# Edit pubspec.yaml and add sherpa_onnx to dependencies

dart pub get
dart run
```
