### use the test wav files to test the model:

./build/bin/Release/sherpa-onnx-offline --tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt --paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/0.wav ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/1.wav ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/8k.wav ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/2-zh-en.wav ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/3-sichuan.wav ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/4-tianjin.wav ./sherpa-onnx-paraformer-zh-small-2024-03-09/test_wavs/5-henan.wav

### use the microphone to test the model:

./build/bin/Release/sherpa-onnx-microphone-offline --tokens=./sherpa-onnx-paraformer-zh-small-2024-03-09/tokens.txt --paraformer=./sherpa-onnx-paraformer-zh-small-2024-03-09/model.int8.onnx

### In Windows, if the output is wired, you can try to use the following command to test the model:

[Frequently Asked Question (FAQs) — sherpa 1.3 documentation](https://k2-fsa.github.io/sherpa/onnx/tts/faq.html) & command:

```
CHCP65001
```
