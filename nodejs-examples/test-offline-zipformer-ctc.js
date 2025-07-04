// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let config = {
    modelConfig: {
      zipformerCtc: {
        model: './sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx',
      },
      tokens: './sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt',
    }
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
