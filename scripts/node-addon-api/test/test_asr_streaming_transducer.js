// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('../lib/sherpa-onnx.js');
const performance = require('perf_hooks').performance;

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'transducer': {
      'encoder':
          './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx',
      'decoder':
          './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx',
      'joiner':
          './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx',
    },
    'tokens':
        './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
    'modelType': 'zipformer',
  }
};

const waveFilename =
    './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav';

const recognizer = new sherpa_onnx.OnlineRecognizer(config);
console.log('Started')
let start = performance.now();
const stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({samples: wave.samples, sampleRate: wave.sampleRate});

const tailPadding = new Float32Array(wave.sampleRate * 0.4);
stream.acceptWaveform({samples: tailPadding, sampleRate: wave.sampleRate});

while (recognizer.isReady(stream)) {
  recognizer.decode(stream);
}
result = recognizer.getResult(stream)
let stop = performance.now();
console.log('Done')

const elapsed_seconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'secodns')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'secodns')
console.log('RTF', real_time_factor.toFixed(3))
console.log('result', result.text)
