// Copyright (c)  2026  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'cohereTranscribe': {
      'encoder':
          './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx',
      'decoder':
          './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx',
      'usePunct': 1,
      'useItn': 1,
    },
    'tokens':
        './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 0,
  }
};

const waveFilename =
    './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
const stream = recognizer.createStream();
stream.setOption('language', 'en');
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

recognizer.decode(stream);
const result = recognizer.getResult(stream);
console.log('result\n', result);
