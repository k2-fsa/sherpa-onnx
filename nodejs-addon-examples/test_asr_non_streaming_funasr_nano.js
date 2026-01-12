// Copyright (c)  2026  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'funasrNano': {
      'encoderAdaptor':
          './sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx',
      'llm': './sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx',
      'embedding':
          './sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx',
      'tokenizer': './sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B',
    },
    'tokens': '',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
  }
};

const waveFilename =
    './sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
console.log('Started')
let start = Date.now();
const stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

recognizer.decode(stream);
const result = recognizer.getResult(stream);
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
