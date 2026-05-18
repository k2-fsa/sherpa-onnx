// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)

const cpal = require('node-cpal');

const sherpa_onnx = require('sherpa-onnx-node');

function createVad() {
  // please download silero_vad.onnx from
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  //
  // OR
  //
  // please download ten-vad.onnx from
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx
  const config = {
    sileroVad: {
      // model: '',
      model: './silero_vad.onnx',
      threshold: 0.5,
      minSpeechDuration: 0.25,
      minSilenceDuration: 0.5,
      windowSize: 512,
    },
    tenVad: {
      model: '',
      // model: './ten-vad.onnx',
      threshold: 0.5,
      minSpeechDuration: 0.25,
      minSilenceDuration: 0.5,
      windowSize: 256,
    },
    sampleRate: 16000,
    debug: true,
    numThreads: 1,
  };

  const bufferSizeInSeconds = 60;

  return new sherpa_onnx.Vad(config, bufferSizeInSeconds);
}

const vad = createVad();

const bufferSizeInSeconds = 30;
const buffer =
    new sherpa_onnx.CircularBuffer(bufferSizeInSeconds * vad.config.sampleRate);

const inputDevice = cpal.getDefaultInputDevice();
const deviceConfig = cpal.getDefaultInputConfig(inputDevice.deviceId);
const nativeSampleRate = deviceConfig.sampleRate;
const targetSampleRate = vad.config.sampleRate;
const resampler =
    new sherpa_onnx.LinearResampler(nativeSampleRate, targetSampleRate);

console.log(
    `Device: ${inputDevice.name}, native sample rate: ${nativeSampleRate} Hz`);
console.log(`Resampling from ${nativeSampleRate} to ${targetSampleRate} Hz`);

let printed = false;
let index = 0;

const inputStream = cpal.createStream(
    inputDevice.deviceId,
    true,  // true = input/recording stream
    {
      sampleRate: nativeSampleRate,
      channels: 1,
      format: 'f32',
    },
    (data) => {
      const windowSize = vad.config.sileroVad.model != '' ?
          vad.config.sileroVad.windowSize :
          vad.config.tenVad.windowSize;

      const resampled = resampler.resample(data);
      buffer.push(resampled);
      while (buffer.size() >= windowSize) {
        const samples = buffer.get(buffer.head(), windowSize);
        buffer.pop(windowSize);
        vad.acceptWaveform(samples);
        if (vad.isDetected() && !printed) {
          console.log(`${index}: Detected speech`);
          printed = true;
        }

        if (!vad.isDetected()) {
          printed = false;
        }

        while (!vad.isEmpty()) {
          const segment = vad.front();
          vad.pop();
          const filename = `${index}-${
                               new Date()
                                   .toLocaleTimeString('en-US', {hour12: false})
                                   .split(' ')[0]}.wav`
                               .replace(/:/g, '-');
          sherpa_onnx.writeWave(
              filename,
              {samples: segment.samples, sampleRate: vad.config.sampleRate});
          const duration = segment.samples.length / vad.config.sampleRate;
          console.log(`${index} End of speech. Duration: ${duration} seconds`);
          console.log(`Saved to ${filename}`);
          index += 1;
        }
      }
    });

console.log('Started! Please speak');

process.on('SIGINT', () => {
  cpal.closeStream(inputStream);
  console.log('Free resources');
  process.exit(0);
});
