// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)
//
const cpal = require('node-cpal');


const sherpa_onnx = require('sherpa-onnx-node');

function createOnlineRecognizer() {
  const config = {
    'featConfig': {
      'sampleRate': 16000,
      'featureDim': 80,
    },
    'modelConfig': {
      'paraformer': {
        'encoder':
            './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx',
        'decoder':
            './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx',
      },
      'tokens': './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt',
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

const inputDevice = cpal.getDefaultInputDevice();
const deviceConfig = cpal.getDefaultInputConfig(inputDevice.deviceId);
const nativeSampleRate = deviceConfig.sampleRate;
const targetSampleRate = recognizer.config.featConfig.sampleRate;

const resampler = new sherpa_onnx.LinearResampler(nativeSampleRate, targetSampleRate);
const display = new sherpa_onnx.Display(50);

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
      stream.acceptWaveform(
          {sampleRate: targetSampleRate, samples: resampled});

      while (recognizer.isReady(stream)) {
        recognizer.decode(stream);
      }

      const isEndpoint = recognizer.isEndpoint(stream);
      let text = recognizer.getResult(stream).text.toLowerCase();

      if (isEndpoint) {
        // for online paraformer models, we have to manually padding on endpoint
        // so that the last word can be recognized
        const tailPadding =
            new Float32Array(targetSampleRate * 0.4);
        stream.acceptWaveform({
          samples: tailPadding,
          sampleRate: targetSampleRate
        });
        while (recognizer.isReady(stream)) {
          recognizer.decode(stream);
        }
        text = recognizer.getResult(stream).text.toLowerCase();
      }

      if (text.length > 0 && lastText != text) {
        lastText = text;
        display.print(segmentIndex, lastText);
      }
      if (isEndpoint) {
        if (text.length > 0) {
          lastText = text;
          segmentIndex += 1;
        }
        recognizer.reset(stream);
      }
    });

console.log('Started! Please speak');
