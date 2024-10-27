// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'moonshine': {
      'preprocessor': './sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx',
      'encoder': './sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx',
      'uncachedDecoder':
          './sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx',
      'cachedDecoder':
          './sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx',
    },
    'tokens': './sherpa-onnx-moonshine-tiny-en-int8/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
  }
};

const waveFilename = './sherpa-onnx-moonshine-tiny-en-int8/test_wavs/0.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
console.log('Started')
let start = Date.now();
const stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

recognizer.decode(stream);
result = recognizer.getResult(stream)
let stop = Date.now();
console.log('Done')

const elapsed_seconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'secodns')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'secodns')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))
console.log(waveFilename)
console.log('result\n', result)
