# Speech Enhancement Example

This example shows how to use the Dart offline speech denoiser API with GTCRN
models.

Download GTCRN models and test wave files from:

- https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

Then run:

```bash
dart pub get
dart run ./bin/speech_enhancement_gtcrn.dart --model ./gtcrn_simple.onnx --input-wav ./inp_16k.wav --output-wav ./enhanced-16k.wav
```
