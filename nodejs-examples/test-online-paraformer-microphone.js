// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const portAudio = require('naudiodon2');
console.log(portAudio.getDevices());

const sherpa_onnx = require('sherpa-onnx');

function createOnlineRecognizer() {
  let onlineTransducerModelConfig = {
    encoder: '',
    decoder: '',
    joiner: '',
  };

  let onlineParaformerModelConfig = {
    encoder:
        './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx',
    decoder:
        './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx',
  };

  let onlineZipformer2CtcModelConfig = {
    model: '',
  };

  let onlineModelConfig = {
    transducer: onlineTransducerModelConfig,
    paraformer: onlineParaformerModelConfig,
    zipformer2Ctc: onlineZipformer2CtcModelConfig,
    tokens: './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt',
    numThreads: 1,
    provider: 'cpu',
    debug: 1,
    modelType: 'paraformer',
  };

  let featureConfig = {
    sampleRate: 16000,
    featureDim: 80,
  };

  let recognizerConfig = {
    featConfig: featureConfig,
    modelConfig: onlineModelConfig,
    decodingMethod: 'greedy_search',
    maxActivePaths: 4,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 2.4,
    rule2MinTrailingSilence: 1.2,
    rule3MinUtteranceLength: 20,
    hotwordsFile: '',
    hotwordsScore: 1.5,
    ctcFstDecoderConfig: {
      graph: '',
      maxActive: 3000,
    }
  };

  return sherpa_onnx.createOnlineRecognizer(recognizerConfig);
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

ai.on('data', data => {
  const samples = new Float32Array(data.buffer);

  stream.acceptWaveform(recognizer.config.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  const isEndpoint = recognizer.isEndpoint(stream);
  const text = recognizer.getResult(stream).text;

  if (text.length > 0 && lastText != text) {
    lastText = text;
    console.log(segmentIndex, lastText);
  }
  if (isEndpoint) {
    if (text.length > 0) {
      lastText = text;
      segmentIndex += 1;
    }
    recognizer.reset(stream)
  }
});

ai.on('close', () => {
  console.log('Free resources');
  stream.free();
  recognizer.free();
});

ai.start();
console.log('Started! Please speak')
