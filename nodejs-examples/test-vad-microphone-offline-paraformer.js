// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('sherpa-onnx3');
const portAudio = require('naudiodon2');
console.log(portAudio.getDevices());

function createOfflineRecognizer() {
  const featConfig = new sherpa_onnx.FeatureConfig();
  featConfig.sampleRate = 16000;
  featConfig.featureDim = 80;

  // test online recognizer
  const paraformer = new sherpa_onnx.OfflineParaformerModelConfig();
  paraformer.model = './sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx';
  const tokens = './sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt';

  const modelConfig = new sherpa_onnx.OfflineModelConfig();
  modelConfig.paraformer = paraformer;
  modelConfig.tokens = tokens;
  modelConfig.modelType = 'paraformer';

  const recognizerConfig = new sherpa_onnx.OfflineRecognizerConfig();
  recognizerConfig.featConfig = featConfig;
  recognizerConfig.modelConfig = modelConfig;
  recognizerConfig.decodingMethod = 'greedy_search';

  const recognizer = new sherpa_onnx.OfflineRecognizer(recognizerConfig);
  return recognizer
}

function createVad() {
  const sileroVadModelConfig = new sherpa_onnx.SileroVadModelConfig();
  sileroVadModelConfig.model = './silero_vad.onnx';
  sileroVadModelConfig.minSpeechDuration = 0.3;   // seconds
  sileroVadModelConfig.minSilenceDuration = 0.3;  // seconds
  sileroVadModelConfig.windowSize = 512;

  const vadModelConfig = new sherpa_onnx.VadModelConfig();
  vadModelConfig.sileroVad = sileroVadModelConfig;
  vadModelConfig.sampleRate = 16000;

  const bufferSizeInSeconds = 60;
  const vad = new sherpa_onnx.VoiceActivityDetector(
      vadModelConfig, bufferSizeInSeconds);
  return vad;
}

const recognizer = createOfflineRecognizer();
const vad = createVad();

const bufferSizeInSeconds = 30;
const buffer =
    new sherpa_onnx.CircularBuffer(bufferSizeInSeconds * vad.config.sampleRate);

var ai = new portAudio.AudioIO({
  inOptions: {
    channelCount: 1,
    sampleFormat: portAudio.SampleFormatFloat32,
    sampleRate: vad.config.sampleRate,
    deviceId: -1,  // Use -1 or omit the deviceId to select the default device
    closeOnError: true  // Close the stream if an audio error is detected, if
                        // set false then just log the error
  }
});

let printed = false;
let index = 0;
ai.on('data', data => {
  const windowSize = vad.config.sileroVad.windowSize;
  buffer.push(new Float32Array(data.buffer));
  while (buffer.size() > windowSize) {
    const samples = buffer.get(buffer.head(), windowSize);
    buffer.pop(windowSize);
    vad.acceptWaveform(samples)
  }

  while (!vad.isEmpty()) {
    const segment = vad.front();
    vad.pop();
    const stream = recognizer.createStream();
    stream.acceptWaveform(
        recognizer.config.featConfig.sampleRate, segment.samples);
    recognizer.decode(stream);
    const r = recognizer.getResult(stream);
    stream.free();
    if (r.text.length > 0) {
      console.log(`${index}: ${r.text}`);
      index += 1;
    }
  }
});

ai.on('close', () => {
  console.log('Free resources');
  recognizer.free();
  vad.free();
  buffer.free();
});

ai.start();
console.log('Started! Please speak')
