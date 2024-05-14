// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
//
const portAudio = require('naudiodon2');
// console.log(portAudio.getDevices());

const sherpa_onnx = require('sherpa-onnx-node');

function createOnlineRecognizer() {
  const config = {
    'featConfig': {
      'sampleRate': 16000,
      'featureDim': 80,
    },
    'modelConfig': {
      'zipformer2Ctc': {
        'model':
            './sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx',
      },
      'tokens':
          './sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt',
      'numThreads': 2,
      'provider': 'cpu',
      'debug': 1,
    },
    'decodingMethod': 'greedy_search',
    'maxActivePaths': 4,
    'enableEndpoint': true,
    'rule1MinTrailingSilence': 2.4,
    'rule2MinTrailingSilence': 1.2,
    'rule3MinUtteranceLength': 20
  };

  return new sherpa_onnx.OnlineRecognizer(config);
}

const recognizer = createOnlineRecognizer();
const stream = recognizer.createStream();

let lastText = '';
let segmentIndex = 0;

const ai = new portAudio.AudioIO({
  inOptions: {
    channelCount: 1,
    closeOnError: true,  // Close the stream if an audio error is detected, if
                         // set false then just log the error
    deviceId: -1,  // Use -1 or omit the deviceId to select the default device
    sampleFormat: portAudio.SampleFormatFloat32,
    sampleRate: recognizer.config.featConfig.sampleRate
  }
});

const display = new sherpa_onnx.Display(50);

ai.on('data', data => {
  const samples = new Float32Array(data.buffer);

  stream.acceptWaveform(
      {sampleRate: recognizer.config.featConfig.sampleRate, samples: samples});

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  const isEndpoint = recognizer.isEndpoint(stream);
  const text = recognizer.getResult(stream).text.toLowerCase();

  if (text.length > 0 && lastText != text) {
    lastText = text;
    display.print(segmentIndex, lastText);
  }
  if (isEndpoint) {
    if (text.length > 0) {
      lastText = text;
      segmentIndex += 1;
    }
    recognizer.reset(stream)
  }
});


ai.start();
console.log('Started! Please speak')
