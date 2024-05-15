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
    'paraformer': {
      'encoder':
          './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx',
      'decoder':
          './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx',
    },
    'tokens': './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
  }
};

const waveFilename =
    './sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav';

const recognizer = new sherpa_onnx.OnlineRecognizer(config);
console.log('Started')
let start = Date.now();
const stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

const tailPadding = new Float32Array(wave.sampleRate * 0.4);
stream.acceptWaveform({samples: tailPadding, sampleRate: wave.sampleRate});

while (recognizer.isReady(stream)) {
  recognizer.decode(stream);
}
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
