// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
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
