# Introduction

This directory contains examples for the JAVA API of sherpa-onnx.

# Usage

## Non-streaming speaker diarization

```bash
./run-offline-speaker-diarization.sh
```

## Streaming Speech recognition

```
./run-streaming-asr-from-mic-transducer.sh
./run-streaming-decode-file-ctc.sh
./run-streaming-decode-file-ctc-hlg.sh
./run-streaming-decode-file-paraformer.sh
./run-streaming-decode-file-transducer.sh
```

## Non-Streaming Speech recognition

```bash
./run-non-streaming-decode-file-paraformer.sh
./run-non-streaming-decode-file-sense-voice.sh
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

## Non-Streaming text-to-speech (Play as it is generating)

```bash
./run-non-streaming-tts-piper-en-with-callback.sh
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

## VAD with a microphone

```bash
./run-vad-from-mic.sh
```

## VAD with a microphone + Non-streaming SenseVoice for speech recognition

```bash
./run-vad-from-mic-non-streaming-sense-voice.sh
```

## VAD with a microphone + Non-streaming Paraformer for speech recognition

```bash
./run-vad-from-mic-non-streaming-paraformer.sh
```

## VAD with a microphone + Non-streaming Whisper tiny.en for speech recognition

```bash
./run-vad-from-mic-non-streaming-whisper.sh
```

## VAD (Remove silence)

```bash
./run-vad-remove-slience.sh
```

## VAD + Non-streaming SenseVoice for speech recognition

```bash
./run-vad-non-streaming-sense-voice.sh
```

## VAD + Non-streaming Paraformer for speech recognition

```bash
./run-vad-non-streaming-paraformer.sh
```

## Keyword spotter

```bash
./run-kws-from-file.sh
```
