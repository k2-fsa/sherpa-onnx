// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
//
const cpal = require('node-cpal');


const sherpa_onnx = require('sherpa-onnx-node');

function createRecognizer() {
  // Please download test files from
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  const config = {
    'featConfig': {
      'sampleRate': 16000,
      'featureDim': 80,
    },
    'modelConfig': {
      'transducer': {
        'encoder':
            './sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.int8.onnx',
        'decoder':
            './sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx',
        'joiner':
            './sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.int8.onnx',
      },
      'tokens': './sherpa-onnx-zipformer-en-2023-04-01/tokens.txt',
      'numThreads': 2,
      'provider': 'cpu',
      'debug': 1,
    }
  };

  return new sherpa_onnx.OfflineRecognizer(config);
}

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

const recognizer = createRecognizer();
const vad = createVad();

const bufferSizeInSeconds = 30;
const buffer =
    new sherpa_onnx.CircularBuffer(bufferSizeInSeconds * vad.config.sampleRate);

const inputDevice = cpal.getDefaultInputDevice();
const deviceConfig = cpal.getDefaultInputConfig(inputDevice.deviceId);
const nativeSampleRate = deviceConfig.sampleRate;
const targetSampleRate = vad.config.sampleRate;

const resampler = new sherpa_onnx.LinearResampler(nativeSampleRate, targetSampleRate);

let printed = false;
let index = 0;

const inputStream = cpal.createStream(
    inputDevice.deviceId,
    true,
    {
      sampleRate: nativeSampleRate,
      channels: 1,
      format: 'f32',
    },
    (data) => {
      const resampled = resampler.resample(data);
      const windowSize = vad.config.sileroVad.windowSize;
      buffer.push(resampled);
      while (buffer.size() >= windowSize) {
        const samples = buffer.get(buffer.head(), windowSize);
        buffer.pop(windowSize);
        vad.acceptWaveform(samples);
      }

      while (!vad.isEmpty()) {
        const segment = vad.front();
        vad.pop();
        const stream = recognizer.createStream();
        stream.acceptWaveform({
          samples: segment.samples,
          sampleRate: recognizer.config.featConfig.sampleRate
        });
        recognizer.decode(stream);
        const r = recognizer.getResult(stream);
        if (r.text.length > 0) {
          const text = r.text.toLowerCase().trim();
          console.log(`${index}: ${text}`);

          const filename = `${index}-${text}-${
                               new Date()
                                   .toLocaleTimeString('en-US', {hour12: false})
                                   .split(' ')[0]}.wav`
                               .replace(/:/g, '-');

          sherpa_onnx.writeWave(
              filename,
              {samples: segment.samples, sampleRate: vad.config.sampleRate});

          index += 1;
        }
      }
    });

console.log('Started! Please speak');
