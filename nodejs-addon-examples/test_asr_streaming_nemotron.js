// Copyright (c)  2026  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// This example shows streaming speech recognition with a multilingual
// Nemotron transducer model and per-stream language pinning.
//
// Multilingual Nemotron models read the option 'language' from each stream
// on every decode call, so streams of the same recognizer can be pinned to
// different languages, and a stream can even switch language mid-session.
// Leaving the option unset selects automatic language detection.
// English-only Nemotron models ignore the option.
//
// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

const modelDir =
    './sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11';

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 128,
  },
  'modelConfig': {
    'transducer': {
      'encoder': `${modelDir}/encoder.int8.onnx`,
      'decoder': `${modelDir}/decoder.int8.onnx`,
      'joiner': `${modelDir}/joiner.int8.onnx`,
    },
    'tokens': `${modelDir}/tokens.txt`,
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
  }
};

const recognizer = new sherpa_onnx.OnlineRecognizer(config);

function decode(waveFilename, language) {
  const stream = recognizer.createStream();
  if (language !== undefined) {
    stream.setOption('language', language);
  }
  console.log(
      `hasOption('language'): ${stream.hasOption('language')},`,
      `getOption('language'): '${stream.getOption('language')}'`);

  const wave = sherpa_onnx.readWave(waveFilename);
  stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

  const tailPadding = new Float32Array(wave.sampleRate * 0.4);
  stream.acceptWaveform({samples: tailPadding, sampleRate: wave.sampleRate});
  stream.inputFinished();

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }
  return recognizer.getResult(stream).text;
}

// Streams of the same recognizer, each pinned to its own language.
console.log('de pinned:', decode(`${modelDir}/test_wavs/de.wav`, 'de'));
console.log('es pinned:', decode(`${modelDir}/test_wavs/es.wav`, 'es'));

// Leaving the option unset selects automatic language detection.
console.log('auto:', decode(`${modelDir}/test_wavs/de.wav`));
