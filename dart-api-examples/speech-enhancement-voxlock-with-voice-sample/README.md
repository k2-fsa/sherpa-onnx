# Speech Enhancement Example (VoxLock, with a voice sample)

This example shows how to use the Dart offline speech denoiser API with
VoxLock models and a voice sample.

VoxLock is a target-speaker voice isolation model. It keeps the voice of one
speaker, e.g., the caller in a phone call, and removes the other voices.

The model is from https://github.com/arbazsiddiqui/voxlock

This example provides a voice sample of the speaker to keep; see
../speech-enhancement-voxlock if you don't have one.

Download the model and test wave files from:

- https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

Then run:

```bash
dart pub get
dart run ./bin/speech_enhancement_voxlock_with_voice_sample.dart \
  --enroll-model ./sherpa-onnx-voxlock-int8/enroll.onnx \
  --chunk-model ./sherpa-onnx-voxlock-int8/chunk_int8.onnx \
  --voice-sample ./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav \
  --input-wav ./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav \
  --output-wav ./voxlock-enhanced-with-voice-sample-16k.wav
```
