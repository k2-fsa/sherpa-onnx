# Introduction

This folder contains Go API examples for [sherpa-onnx][sherpa-onnx].

Please refer to the documentation
https://k2-fsa.github.io/sherpa/onnx/go-api/index.html
for details.

- [./non-streaming-decode-files](./non-streaming-decode-files) It shows how to use
  a non-streaming ASR model to decode files

- [./non-streaming-tts](./non-streaming-tts) It shows how to use a non-streaming TTS
  model to convert text to speech

- [./real-time-speech-recognition-from-microphone](./real-time-speech-recognition-from-microphone)
  It shows how to use a streaming ASR model to recognize speech from a microphone in real-time

- [./vad](./vad) It shows how to use silero VAD with Golang.

- [./vad-asr-whisper](./vad-asr-whisper) It shows how to use silero VAD + Whisper
  for speech recognition.

- [./vad-asr-paraformer](./vad-asr-paraformer) It shows how to use silero VAD + Paraformer
  for speech recognition.

- [./vad-spoken-language-identification](./vad-spoken-language-identification) It shows how to use silero VAD + Whisper
  for spoken language identification.

- [./speaker-identification](./speaker-identification) It shows how to use Go API for speaker identification.

- [./vad-speaker-identification](./vad-speaker-identification) It shows how to use Go API for VAD + speaker identification.

[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
