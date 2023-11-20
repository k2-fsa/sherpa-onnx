// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('./index.js');
var portAudio = require('naudiodon2');
console.log(portAudio.getDevices());

let sileroVadModelConfig = new sherpa_onnx.SileroVadModelConfig();
sileroVadModelConfig.model = './silero_vad.onnx';
sileroVadModelConfig.windowSize = 512;

let vadModelConfig = new sherpa_onnx.VadModelConfig();
vadModelConfig.sileroVad = sileroVadModelConfig;
vadModelConfig.sampleRate = 16000;

let bufferSizeInSeconds = 60;
let vad =
    new sherpa_onnx.VoiceActivityDetector(vadModelConfig, bufferSizeInSeconds);
let buffer = new sherpa_onnx.CircularBuffer(bufferSizeInSeconds);

var ai = new portAudio.AudioIO({
  inOptions: {
    channelCount: 1,
    sampleFormat: portAudio.SampleFormatFloat32,
    sampleRate: featConfig.sampleRate,
    deviceId: -1,  // Use -1 or omit the deviceId to select the default device
    closeOnError: true  // Close the stream if an audio error is detected, if
                        // set false then just log the error
  }
});

ai.on('data', data => {
  let samples = new Float32Array(data.buffer);
  buffer.push(samples);
  while (buffer.size() > sileroVadModelConfig.windowSize) {
  }

  stream.acceptWaveform(recognizerConfig.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  let isEndpoint = recognizer.isEndpoint(stream);
  let text = recognizer.getResult(stream).text;

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
