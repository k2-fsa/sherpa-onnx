// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let modelConfig = {
    whisper: {
      encoder: './sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx',
      decoder: './sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx',
      language: '',
      task: 'transcribe',
      tailPaddings: -1,
    },
    tokens: './sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt',
  };

  let config = {
    modelConfig: modelConfig,
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

recognizer = createOfflineRecognizer();
stream = recognizer.createStream();

const waveFilename = './sherpa-onnx-whisper-tiny.en/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
