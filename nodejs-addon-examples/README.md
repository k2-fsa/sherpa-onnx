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

## Streaming speech recognition with Zipformer transducer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

node ./test_asr_streaming_transducer.js

# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_asr_streaming_transducer_microphone.js
```

## Streaming speech recognition with Zipformer CTC

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

## Streaming speech recognition with Paraformer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

node ./test_asr_streaming_paraformer.js

# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_asr_streaming_paraformer_microphone.js
```

## Non-streaming speech recognition with Zipformer transducer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2

node ./test_asr_non_streaming_transducer.js

# To run VAD + non-streaming ASR with transudcer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_transducer_microphone.js
```

## Non-streaming speech recognition with Whisper

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

node ./test_asr_non_streaming_whisper.js

# To run VAD + non-streaming ASR with Paraformer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_whisper_microphone.js
```

## Non-streaming speech recognition with NeMo CTC models

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
tar xvf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
rm sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

node ./test_asr_non_streaming_nemo_ctc.js

# To run VAD + non-streaming ASR with Paraformer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_nemo_ctc_microphone.js
```

## Non-streaming speech recognition with Paraformer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

node ./test_asr_non_streaming_paraformer.js

# To run VAD + non-streaming ASR with Paraformer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_paraformer_microphone.js
```

## Text-to-speech with piper VITS models (TTS)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2
tar xvf vits-piper-en_GB-cori-medium.tar.bz2
rm vits-piper-en_GB-cori-medium.tar.bz2

node ./test_tts_non_streaming_vits_piper_en.js
```

## Text-to-speech with piper Coqui-ai/TTS models (TTS)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
tar xvf vits-coqui-de-css10.tar.bz2
rm vits-coqui-de-css10.tar.bz2

node ./test_tts_non_streaming_vits_coqui_de.js
```

## Text-to-speech with vits Chinese models (1/2)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
tar xvf sherpa-onnx-vits-zh-ll.tar.bz2
rm sherpa-onnx-vits-zh-ll.tar.bz2

node ./test_tts_non_streaming_vits_zh_ll.js
```

## Text-to-speech with vits Chinese models (2/2)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
rm vits-icefall-zh-aishell3.tar.bz2

node ./test_tts_non_streaming_vits_zh_aishell3.js
```
