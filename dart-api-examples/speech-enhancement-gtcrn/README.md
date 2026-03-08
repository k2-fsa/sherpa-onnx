# Speech Enhancement Example

This example shows how to use the Dart offline speech denoiser API with GTCRN
or DPDFNet models.

Use 16 kHz DPDFNet models such as `baseline.onnx`, `dpdfnet2.onnx`,
`dpdfnet4.onnx`, or `dpdfnet8.onnx` for downstream ASR or speech recognition.
Use `dpdfnet2_48khz_hr.onnx` for 48 kHz enhancement output.

Download a model and a test wave file from:

https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

Then run:

```bash
dart pub get
dart run ./bin/speech_enhancement_gtcrn.dart --model ./gtcrn_simple.onnx
```
