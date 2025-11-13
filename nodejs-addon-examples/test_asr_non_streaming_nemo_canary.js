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
    'canary': {
      'encoder':
          './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx',
      'decoder':
          './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx',
      'srcLang': 'en',
      'tgtLang': 'en',
      'usePnc': 1,
    },
    'tokens':
        './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 0,
  }
};

const waveFilename =
    './sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
console.log('Started')
let start = Date.now();
let stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

recognizer.decode(stream);
let result = recognizer.getResult(stream);
let stop = Date.now();
console.log('Done')

const elapsed_seconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))
console.log(waveFilename)
console.log('result (English)\n', result)

stream = recognizer.createStream();
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});
recognizer.config.modelConfig.canary.tgtLang = 'de';
recognizer.setConfig(recognizer.config);

recognizer.decode(stream);
result = recognizer.getResult(stream)
console.log('result (German)\n', result)
