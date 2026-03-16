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
  It shows how to use the offline speech denoiser API with GTCRN or DPDFNet
  models. Use 16 kHz DPDFNet models such as `baseline.onnx`,
  `dpdfnet2.onnx`, `dpdfnet4.onnx`, or `dpdfnet8.onnx` for downstream ASR and
  `dpdfnet2_48khz_hr.onnx` for 48 kHz enhancement output.
