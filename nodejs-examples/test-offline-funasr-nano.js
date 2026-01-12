// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
/*
This script throws the following error since the model is too large
for WebAssembly to run it.

wasm://wasm/0308b752:1
RuntimeError: memory access out of bounds
    at wasm://wasm/0308b752:wasm-function[66]:0xe3ca
    at wasm://wasm/0308b752:wasm-function[134]:0x1464f
    at wasm://wasm/0308b752:wasm-function[4492]:0x34e46b
    at wasm://wasm/0308b752:wasm-function[9931]:0x6c8fc1
    at new OfflineRecognizer (/home/runner/work/sherpa-onnx/sherpa-onnx/scripts/nodejs/sherpa-onnx-asr.js:1530:27)
    at Object.createOfflineRecognizer (/home/runner/work/sherpa-onnx/sherpa-onnx/scripts/nodejs/index.js:22:10)
    at createOfflineRecognizer (/home/runner/work/sherpa-onnx/sherpa-onnx/scripts/nodejs/test-offline-funasr-nano.js:24:22)
    at Object.<anonymous> (/home/runner/work/sherpa-onnx/sherpa-onnx/scripts/nodejs/test-offline-funasr-nano.js:27:20)
    at Module._compile (node:internal/modules/cjs/loader:1521:14)
    at Module._extensions..js (node:internal/modules/cjs/loader:1623:10)
Node.js v20.19.6

 */
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let config = {
    modelConfig: {
      funasrNano: {
        encoderAdaptor:
            './sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx',
        llm: './sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx',
        embedding:
            './sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx',
        tokenizer: './sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B',
      },
      tokens: '',
    }
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
