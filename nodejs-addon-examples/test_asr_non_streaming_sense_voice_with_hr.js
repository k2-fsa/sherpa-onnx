// Copyright (c)  2025  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files


// If your path contains non-ascii characters, e.g., Chinese, you can use
// the following code
//

// let encoder = new TextEncoder();
// let tokens = encoder.encode(
//     './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/测试.txt');
// let model = encoder.encode(
//     './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/测试.int8.onnx');


const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'senseVoice': {
      'model':
          './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx',
      // 'model': model,
      'useInverseTextNormalization': 1,
    },
    'tokens': './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt',
    // 'tokens': tokens,
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
  },
  'hr': {
    // Please download files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files
    'lexicon': './lexicon.txt',
    'ruleFsts': './replace.fst',
  }
};

const waveFilename = './test-hr.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
console.log('Started')
let start = Date.now();
const stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

recognizer.decode(stream);
const result = recognizer.getResult(stream)
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
console.log('result\n', result)
