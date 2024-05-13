# Introduction

Note: You need `Node >= 10`.

This repo contains examples for NodeJS.
It uses [node-addon-api](https://github.com/nodejs/node-addon-api) to wrap
`sherpa-onnx` for NodeJS and it supports multiple threads.

Note: [../nodejs-examples](../nodejs-examples) uses WebAssembly to wrap
`sherpa-onnx` for NodeJS and it does not support multiple threads.

Before you continue, please first run

```bash
npm install

# For macOS x64
export DYLD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-darwin-x64:$DYLD_LIBRARY_PATH

# For macOS arm64
export DYLD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-darwin-arm64:$DYLD_LIBRARY_PATH

# For Linux x64
export LD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-linux-x64:$LD_LIBRARY_PATH

# For Linux arm64, e.g., Raspberry Pi 4
export LD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-linux-arm64:$LD_LIBRARY_PATH
```

# Voice Activity detection (VAD)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx


# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_vad_microphone.js
```

## Streaming speech recognition with zipformer transducer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

node ./test_asr_streaming_transducer.js

# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_asr_streaming_transducer_microphone.js
```

## Streaming speech recognition with zipformer CTC

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2

node ./test_asr_streaming_ctc.js

# To decode with HLG.fst
node ./test_asr_streaming_ctc_hlg.js

# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_asr_streaming_ctc_microphone.js
node ./test_asr_streaming_ctc_hlg_microphone.js
```
