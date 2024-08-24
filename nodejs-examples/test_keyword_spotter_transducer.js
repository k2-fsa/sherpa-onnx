// Copyright (c)  2024  Xiaomi Corporation
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createKeywordSpotter() {
  // Please download test files from
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
  const config = {
    'featConfig': {
      'sampleRate': 16000,
      'featureDim': 80,
    },
    'modelConfig': {
      'transducer': {
        'encoder':
            './sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx',
        'decoder':
            './sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
        'joiner':
            './sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx',
      },
      'tokens':
          './sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt',
      'numThreads': 1,
      'provider': 'cpu',
      'debug': 1,
    },
    keywords: 'w én s ēn t è k ǎ s uǒ  @文森特卡索\n' +
        'f ǎ g uó @法国'
  };

  return new sherpa_onnx.createKws(config);
}

const kws = createKeywordSpotter();
const stream = kws.createStream();
const waveFilename =
    './sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav';

const reader = new wav.Reader();
const readable = new Readable().wrap(reader);

function decode(samples) {
  stream.acceptWaveform(kws.config.featConfig.sampleRate, samples);

  while (kws.isReady(stream)) {
    kws.decode(stream);
    // we have to check the result immediately after decoding!
    const keyword = kws.getResult(stream).keyword;
    if (keyword != '') {
      console.log('Detected', keyword);
    }
  }
}

reader.on('format', ({audioFormat, bitDepth, channels, sampleRate}) => {
  if (sampleRate != kws.config.featConfig.sampleRate) {
    throw new Error(`Only support sampleRate ${
        kws.config.featConfig.sampleRate}. Given ${sampleRate}`);
  }

  if (audioFormat != 1) {
    throw new Error(`Only support PCM format. Given ${audioFormat}`);
  }

  if (channels != 1) {
    throw new Error(`Only a single channel. Given ${channel}`);
  }

  if (bitDepth != 16) {
    throw new Error(`Only support 16-bit samples. Given ${bitDepth}`);
  }
});

fs.createReadStream(waveFilename, {'highWaterMark': 4096})
    .pipe(reader)
    .on('finish', function(err) {
      // tail padding
      const floatSamples =
          new Float32Array(kws.config.featConfig.sampleRate * 0.5);
      decode(floatSamples);
      stream.free();
      kws.free();
    });

readable.on('readable', function() {
  let chunk;
  while ((chunk = readable.read()) != null) {
    const int16Samples = new Int16Array(
        chunk.buffer, chunk.byteOffset,
        chunk.length / Int16Array.BYTES_PER_ELEMENT);

    const floatSamples = new Float32Array(int16Samples.length);

    for (let i = 0; i < floatSamples.length; i++) {
      floatSamples[i] = int16Samples[i] / 32768.0;
    }

    decode(floatSamples);
  }
});
