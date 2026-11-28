// Copyright (c)  2026  graehl
//
const fs = require('fs');
const {Readable} = require('stream');

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let config = {
    modelConfig: {
      w2vBertCtc: {
        model: './sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx',
      },
      tokens: './sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt',
    }
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
