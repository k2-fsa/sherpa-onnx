// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let config = {
    modelConfig: {
      canary: {
        encoder:
            './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx',
        decoder:
            './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx',
        srcLang: 'en',
        dstLang: 'en',
        usePnc: 1,
      },
      tokens:
          './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt',
    }
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
let stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
let text = recognizer.getResult(stream).text;
console.log(`text in English: ${text}`);

stream.free();

// now output German text
recognizer.config.modelConfig.canary.tgt_lang = 'de';
recognizer.setConfig(recognizer.config);

stream = recognizer.createStream();
stream.acceptWaveform(wave.sampleRate, wave.samples);
recognizer.decode(stream);
text = recognizer.getResult(stream).text;

console.log(`text in German: ${text}`);

stream.free();
recognizer.free();
