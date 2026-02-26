// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let config = {
    modelConfig: {
      fireRedAsrCtc: {
        model:
            './sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx',
      },
      tokens:
          './sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt',
    }
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/1.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
