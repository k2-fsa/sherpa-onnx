// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const portAudio = require('naudiodon2');
console.log(portAudio.getDevices());

const sherpa_onnx = require('sherpa-onnx');

function createRecognizer() {
  const featConfig = new sherpa_onnx.FeatureConfig();
  featConfig.sampleRate = 16000;
  featConfig.featureDim = 80;

  const paraformer = new sherpa_onnx.OnlineParaformerModelConfig();
  paraformer.encoder =
      './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx';
  paraformer.decoder =
      './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx';
  const tokens =
      './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt';

  const modelConfig = new sherpa_onnx.OnlineModelConfig();
  modelConfig.paraformer = paraformer;
  modelConfig.tokens = tokens;
  modelConfig.modelType = 'paraformer';

  const recognizerConfig = new sherpa_onnx.OnlineRecognizerConfig();
  recognizerConfig.featConfig = featConfig;
  recognizerConfig.modelConfig = modelConfig;
  recognizerConfig.decodingMethod = 'greedy_search';
  recognizerConfig.enableEndpoint = 1;

  const recognizer = new sherpa_onnx.OnlineRecognizer(recognizerConfig);
  return recognizer;
}
recognizer = createRecognizer();
stream = recognizer.createStream();

display = new sherpa_onnx.Display(50);

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

ai.on('close', () => {
  console.log('Free resources');
  stream.free();
  recognizer.free();
});

ai.start();
console.log('Started! Please speak')
