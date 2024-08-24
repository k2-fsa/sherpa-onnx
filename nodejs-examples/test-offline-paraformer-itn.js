// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let modelConfig = {
    paraformer: {
      model: './sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx',
    },
    tokens: './sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt',
  };

  let config = {
    modelConfig: modelConfig,
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
    ruleFsts: './itn_zh_number.fst',
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}


const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

// https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
const waveFilename = './itn-zh-number.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
