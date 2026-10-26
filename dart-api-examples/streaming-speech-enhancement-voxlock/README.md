# Streaming Speech Enhancement Example (VoxLock)

This example shows how to use the Dart online speech denoiser API with VoxLock
models.

VoxLock is a target-speaker voice isolation model. It keeps the voice of one
speaker, e.g., the caller in a phone call, and removes the other voices.

The model is from https://github.com/arbazsiddiqui/voxlock

This example does not provide a voice sample; see
../streaming-speech-enhancement-voxlock-with-voice-sample if you have one.

Download the model and test wave files from:

- https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

Then run:

```bash
dart pub get
dart run ./bin/streaming_speech_enhancement_voxlock.dart \
  --enroll-model ./sherpa-onnx-voxlock-int8/enroll.onnx \
  --chunk-model ./sherpa-onnx-voxlock-int8/chunk_int8.onnx \
  --input-wav ./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav \
  --output-wav ./voxlock-enhanced-online-16k.wav
```
