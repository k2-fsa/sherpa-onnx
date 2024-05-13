// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)

const portAudio = require('naudiodon2');
// console.log(portAudio.getDevices());

const sherpa_onnx = require('sherpa-onnx-node');

function createVad() {
  // please download silero_vad.onnx from
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  const config = {
    sileroVad: {
      model: './silero_vad.onnx',
      threshold: 0.5,
      minSpeechDuration: 0.25,
      minSilenceDuration: 0.5,
      windowSize: 512,
    },
    sampleRate: 16000,
    debug: true,
    numThreads: 1,
  };

  const bufferSizeInSeconds = 60;

  return new sherpa_onnx.Vad(config, bufferSizeInSeconds);
}

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
function createSpokenLanguageID() {
  const config = {
    whisper: {
      encoder: './sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx',
      decoder: './sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx',
    },
    debug: true,
    numThreads: 1,
    provider: 'cpu',
  };
  return new sherpa_onnx.SpokenLanguageIdentification(config);
}

const slid = createSpokenLanguageID();
const vad = createVad();

const display = new Intl.DisplayNames(['en'], {type: 'language'})

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
    sampleRate: vad.config.sampleRate,
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

      const stream = slid.createStream();
      stream.acceptWaveform(
          {samples: segment.samples, sampleRate: vad.config.sampleRate});
      const lang = slid.compute(stream);
      const fullLang = display.of(lang);

      const filename = `${index}-${fullLang}-${
          new Date()
              .toLocaleTimeString('en-US', {hour12: false})
              .split(' ')[0]}.wav`;
      sherpa_onnx.writeWave(
          filename,
          {samples: segment.samples, sampleRate: vad.config.sampleRate});
      const duration = segment.samples.length / vad.config.sampleRate;
      console.log(`${index} End of speech. Duration: ${
          duration} seconds.\n Detected language: ${fullLang}`);
      console.log(`Saved to ${filename}`);
      index += 1;
    }
  }
});

ai.on('close', () => {
  console.log('Free resources');
});

ai.start();
console.log('Started! Please speak')
