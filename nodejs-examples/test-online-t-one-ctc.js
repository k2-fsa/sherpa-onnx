// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createOnlineRecognizer() {
  let toneCtc = {
    model: './sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx',
  };

  let onlineModelConfig = {
    toneCtc: toneCtc,
    tokens: './sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt',
    numThreads: 1,
    provider: 'cpu',
    debug: 1,
  };


  let recognizerConfig = {
    modelConfig: onlineModelConfig,
    decodingMethod: 'greedy_search',
    maxActivePaths: 4,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 2.4,
    rule2MinTrailingSilence: 1.2,
    rule3MinUtteranceLength: 20,
  };

  return sherpa_onnx.createOnlineRecognizer(recognizerConfig);
}

const recognizer = createOnlineRecognizer();
const stream = recognizer.createStream();

const waveFilename = './sherpa-onnx-streaming-t-one-russian-2025-09-08/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);

const leftPadding = new Float32Array(wave.sampleRate * 0.3);
const tailPadding = new Float32Array(wave.sampleRate * 0.6);

stream.acceptWaveform(wave.sampleRate, leftPadding);
stream.acceptWaveform(wave.sampleRate, wave.samples);
stream.acceptWaveform(wave.sampleRate, tailPadding);

while (recognizer.isReady(stream)) {
  recognizer.decode(stream);
}
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
