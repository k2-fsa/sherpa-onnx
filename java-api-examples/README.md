# Introduction

This directory contains examples for the JAVA API of sherpa-onnx.

# Usage

## Streaming Speech recognition

```
./run-streaming-decode-file-ctc.sh
./run-streaming-decode-file-ctc-hlg.sh
./run-streaming-decode-file-paraformer.sh
./run-streaming-decode-file-transducer.sh
```

## Non-Streaming Speech recognition

```bash
./run-non-streaming-decode-file-paraformer.sh
./run-non-streaming-decode-file-transducer.sh
./run-non-streaming-decode-file-whisper.sh
./run-non-streaming-decode-file-nemo.sh
```

## Non-Streaming text-to-speech

```bash
./run-non-streaming-tts-piper-en.sh
./run-non-streaming-tts-coqui-de.sh
./run-non-streaming-tts-vits-zh.sh
```

## Spoken language identification

```bash
./run-spoken-language-identification-whisper.sh
```

## Add punctuations to text

The punctuation model supports both English and Chinese.

```bash
./run-add-punctuation-zh-en.sh
```

## Audio tagging

```bash
./run-audio-tagging-zipformer-from-file.sh
./run-audio-tagging-ced-from-file.sh
```

## Speaker identification

```bash
./run-speaker-identification.sh
```

## VAD (Remove silence)

```bash
./run-vad-remove-slience.sh
```

## VAD + Non-streaming Paraformer for speech recognition

```bash
./run-vad-non-streaming-paraformer.sh
```

## Keyword spotter

```bash
./run-kws-from-file.sh
```
