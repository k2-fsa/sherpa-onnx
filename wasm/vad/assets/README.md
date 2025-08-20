# Introduction

## Use silero-vad

Please download
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
and put `silero_vad.onnx` into the current directory, i.e., `wasm/vad/assets`.

You can find example build script at
https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/wasm-simd-hf-space-silero-vad.yaml

```
cd /path/to/sherpa-onnx/wasm/vad/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

## Use ten-vad

Please download
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
and put `ten-vad.onnx` into the current directory, i.e., `wasm/vad/assets`.

You can find example build script at
https://github.com/k2-fsa/sherpa-onnx/blob/master/.github/workflows/wasm-simd-hf-space-ten-vad.yaml

```
cd /path/to/sherpa-onnx/wasm/vad/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
cd ..
sed -i.bak "s|.*(with <a .*|    (with <a href="https://github.com/TEN-framework/ten-vad">ten-vad</a>)|" ./index.html

```
