// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming Zipformer CTC model with an HLG
// decoding graph via sherpa-onnx's Node-addon API.
//
// The HLG graph is passed to the C API through ctcFstDecoderConfig.
const sherpa_onnx = require('sherpa-onnx-node');

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'zipformerCtc': {
      'model': './sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx',
    },
    'tokens': './sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
  },
  // This is what the example exercises: the graph reaches the C API through
  // ctcFstDecoderConfig.
  'ctcFstDecoderConfig': {
    'graph': './sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst',
    'maxActive': 3000,
  },
};

const waveFilename = './sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
console.log('Started');
let start = Date.now();
const stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

recognizer.decode(stream);
const result = recognizer.getResult(stream);
let stop = Date.now();
console.log('Done');

const elapsed_seconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds');
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3));
console.log(waveFilename);
console.log('result\n', result);
