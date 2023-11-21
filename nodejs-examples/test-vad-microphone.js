// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');
const portAudio = require('naudiodon2');
console.log(portAudio.getDevices());

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
vad = createVad();
const bufferSizeInSeconds = 30;
const buffer =
    new sherpa_onnx.CircularBuffer(bufferSizeInSeconds * vad.config.sampleRate);

const ai = new portAudio.AudioIO({
  inOptions: {
    channelCount: 1,
    closeOnError: true,  // Close the stream if an audio error is detected, if
                         // set false then just log the error
    deviceId: -1,  // Use -1 or omit the deviceId to select the default device
    sampleFormat: portAudio.SampleFormatFloat32,
    sampleRate: vad.config.sampleRate
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
    if (vad.isDetected() && !printed) {
      console.log(`${index}: Detected speech`)
      printed = true;
    }

    if (!vad.isDetected()) {
      printed = false;
    }

    while (!vad.isEmpty()) {
      const segment = vad.front();
      vad.pop();
      const duration = segment.samples.length / vad.config.sampleRate;
      console.log(`${index} End of speech. Duration: ${duration} seconds`);
      index += 1;
    }
  }
});

ai.on('close', () => {
  console.log('Free resources');
  vad.free();
  buffer.free();
});

ai.start();
console.log('Started! Please speak')
