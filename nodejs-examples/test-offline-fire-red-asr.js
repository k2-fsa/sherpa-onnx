// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let modelConfig = {
    fireRedAsr: {
      encoder:
          './sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx',
      decoder:
          './sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx',
    },
    tokens: './sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt',
    debug: 1,
  };

  let config = {
    modelConfig: modelConfig,
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
