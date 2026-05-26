# Speech Enhancement Example

This example shows how to use the Dart offline speech denoiser API with
DPDFNet models.

Use 16 kHz DPDFNet models such as `dpdfnet_baseline.onnx`, `dpdfnet2.onnx`,
`dpdfnet4.onnx`, or `dpdfnet8.onnx` for downstream ASR or speech recognition.
Use `dpdfnet2_48khz_hr.onnx` for 48 kHz enhancement output.

DPDFNet models are available from either:

- https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
- https://huggingface.co/Ceva-IP/DPDFNet

Then run:

```bash
dart pub get
dart run ./bin/speech_enhancement_dpdfnet.dart --model ./dpdfnet_baseline.onnx --input-wav ./inp_16k.wav --output-wav ./enhanced-16k.wav
```
