# Introduction

This folder contains Go API examples for [sherpa-onnx][sherpa-onnx].

Please refer to the documentation
https://k2-fsa.github.io/sherpa/onnx/go-api/index.html
for details.

- [./add-punctuation](./add-punctuation) It shows how to use
  a punctuation model to add punctuations to text

- [./add-punctuation-online](./add-punctuation-online) It shows how to use
  an online punctuation model to add punctuations and casing to text

- [./non-streaming-decode-files](./non-streaming-decode-files) It shows how to use
  a non-streaming ASR model to decode files

- [./non-streaming-speaker-diarization](./non-streaming-speaker-diarization) It shows how to use
  a speaker segmentation model and a speaker embedding model for speaker diarization.

- [./speech-enhancement-gtcrn](./speech-enhancement-gtcrn) It shows how to use
  the offline speech denoiser API with GTCRN models.

- [./speech-enhancement-dpdfnet](./speech-enhancement-dpdfnet) It shows how to use
  the offline speech denoiser API with DPDFNet models.

- [./streaming-speech-enhancement-gtcrn](./streaming-speech-enhancement-gtcrn) It shows how to use
  the online speech denoiser API with GTCRN models.

- [./streaming-speech-enhancement-dpdfnet](./streaming-speech-enhancement-dpdfnet) It shows how to use
  the online speech denoiser API with DPDFNet models.

- [./non-streaming-tts](./non-streaming-tts) It shows how to use a non-streaming TTS
  model to convert text to speech

- [./offline-tts-play](./offline-tts-play) It shows how to use a non-streaming TTS
  model to convert text to speech. It plays the audio back as it is being generated.

- [./zero-shot-pocket-tts](./zero-shot-pocket-tts) It shows how to use a PocketTTS
  model for zero-shot TTS.

- [./zero-shot-pocket-tts-play](./zero-shot-pocket-tts-play) It shows how to use a PocketTTS
  model for zero-shot TTS. It plays the audio back as it is being generated.

- [./zero-shot-zipvoice-tts](./zero-shot-zipvoice-tts) It shows how to use a ZipVoice
  model for zero-shot TTS with the GenerationConfig API.

- [./zero-shot-zipvoice-tts-play](./zero-shot-zipvoice-tts-play) It shows how to use a
  ZipVoice model for zero-shot TTS. It plays the audio back as it is being generated.

- [./real-time-speech-recognition-from-microphone](./real-time-speech-recognition-from-microphone)
  It shows how to use a streaming ASR model to recognize speech from a microphone in real-time

- [./speaker-identification](./speaker-identification) It shows how to use a speaker
  embedding model for speaker identification.

- [./streaming-decode-files](./streaming-decode-files) It shows how to use a streaming
  model for streaming speech recognition

- [./streaming-hlg-decoding](./streaming-hlg-decoding) It shows how to use a streaming
  model for streaming speech recognition with HLG decoding

- [./vad](./vad) It shows how to use silero VAD with Golang.

- [./vad-asr-paraformer](./vad-asr-paraformer) It shows how to use silero VAD + Paraformer
  for speech recognition.

- [./vad-asr-whisper](./vad-asr-whisper) It shows how to use silero VAD + Whisper

- [./vad-speaker-identification](./vad-speaker-identification) It shows how to use Go API for VAD + speaker identification.
  for speech recognition.

- [./vad-spoken-language-identification](./vad-spoken-language-identification) It shows how to use silero VAD + Whisper
  for spoken language identification.

[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
