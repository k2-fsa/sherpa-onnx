// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx');

function createKeywordSpotter() {
  // Please download test files from
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
  const config = {
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
    },
    keywords: 'w én s ēn t è k ǎ s uǒ  @文森特卡索\n' +
        'f ǎ g uó @法国'
  };

  return sherpa_onnx.createKws(config);
}

const kws = createKeywordSpotter();
const stream = kws.createStream();
const waveFilename =
    './sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav';

const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

const tailPadding = new Float32Array(wave.sampleRate * 0.4);
stream.acceptWaveform(kws.config.featConfig.sampleRate, tailPadding);

const detectedKeywords = [];
while (kws.isReady(stream)) {
  kws.decode(stream);
  const keyword = kws.getResult(stream).keyword;
  if (keyword != '') {
    detectedKeywords.push(keyword);
  }
}
console.log(detectedKeywords);

stream.free();
kws.free();
