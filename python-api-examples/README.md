# File description

- [./http_server.py](./http_server.py) It defines which files to server.
  Files are saved in [./web](./web).
- [non_streaming_server.py](./non_streaming_server.py) WebSocket server for
  non-streaming models.
- [vad-remove-non-speech-segments.py](./vad-remove-non-speech-segments.py) It uses
  [silero-vad](https://github.com/snakers4/silero-vad) to remove non-speech
  segments and concatenate all speech segments into a single one.
- [vad-with-non-streaming-asr.py](./vad-with-non-streaming-asr.py) It shows
  how to use VAD with a non-streaming ASR model for speech recognition from
  a microphone
- [offline-speech-enhancement-gtcrn.py](./offline-speech-enhancement-gtcrn.py)
  It shows how to use the offline speech denoiser API with GTCRN.

- [offline-speech-enhancement-dpdfnet.py](./offline-speech-enhancement-dpdfnet.py)
  It shows how to use the offline speech denoiser API with DPDFNet.

- [online-speech-enhancement-gtcrn.py](./online-speech-enhancement-gtcrn.py)
  It shows how to use the online speech denoiser API with GTCRN.

- [online-speech-enhancement-dpdfnet.py](./online-speech-enhancement-dpdfnet.py)
  It shows how to use the online speech denoiser API with DPDFNet.
  models. Use 16 kHz DPDFNet models such as `dpdfnet_baseline.onnx`,
  `dpdfnet2.onnx`, `dpdfnet4.onnx`, or `dpdfnet8.onnx` for downstream ASR and
  `dpdfnet2_48khz_hr.onnx` for 48 kHz enhancement output.

- [pocket-tts.py](./pocket-tts.py) It shows how to use PocketTTS with the
  `GenerationConfig` API.

- [supertonic-tts.py](./supertonic-tts.py) It shows how to use SupertonicTTS
  with the `GenerationConfig` API.

- [zipvoice-tts.py](./zipvoice-tts.py) It shows how to use ZipVoice for
  zero-shot TTS with the `GenerationConfig` API.

- [zipvoice-tts-play.py](./zipvoice-tts-play.py) It shows how to use ZipVoice
  for zero-shot TTS and plays the generated audio while it is being synthesized.
