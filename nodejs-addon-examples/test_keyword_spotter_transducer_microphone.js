// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
//
const portAudio = require('naudiodon2');
// console.log(portAudio.getDevices());

const sherpa_onnx = require('sherpa-onnx-node');

function createKeywordSpotter() {
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
      'numThreads': 2,
      'provider': 'cpu',
      'debug': 1,
    },
    'keywordsFile':
        './sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/keywords.txt',
  };

  return new sherpa_onnx.KeywordSpotter(config);
}

const kws = createKeywordSpotter();
const stream = kws.createStream();

let lastText = '';
let segmentIndex = 0;

const ai = new portAudio.AudioIO({
  inOptions: {
    channelCount: 1,
    closeOnError: true,  // Close the stream if an audio error is detected, if
                         // set false then just log the error
    deviceId: -1,  // Use -1 or omit the deviceId to select the default device
    sampleFormat: portAudio.SampleFormatFloat32,
    sampleRate: kws.config.featConfig.sampleRate
  }
});

const display = new sherpa_onnx.Display(50);

ai.on('data', data => {
  const samples = new Float32Array(data.buffer);

  stream.acceptWaveform(
      {sampleRate: kws.config.featConfig.sampleRate, samples: samples});

  while (kws.isReady(stream)) {
    kws.decode(stream);
  }

  const keyword = kws.getResult(stream).keyword
  if (keyword != '') {
    display.print(segmentIndex, keyword);
    segmentIndex += 1;
  }
});

ai.start();
console.log('Started! Please speak.')
console.log(`Only words from ${kws.config.keywordsFile} can be recognized`)
