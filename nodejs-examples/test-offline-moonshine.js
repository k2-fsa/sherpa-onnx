// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let modelConfig = {
    moonshine: {
      preprocessor: './sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx',
      encoder: './sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx',
      uncachedDecoder:
          './sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx',
      cachedDecoder:
          './sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx',
    },
    tokens: './sherpa-onnx-moonshine-tiny-en-int8/tokens.txt',
  };

  let config = {
    modelConfig: modelConfig,
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

recognizer = createOfflineRecognizer();
stream = recognizer.createStream();

const waveFilename = './sherpa-onnx-moonshine-tiny-en-int8/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
