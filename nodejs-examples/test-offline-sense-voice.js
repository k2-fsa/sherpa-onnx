// Copyright (c)  2024  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let modelConfig = {
    senseVoice: {
      model:
          './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx',
      language: '',
      useInverseTextNormalization: 1,
    },
    tokens: './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt',
  };

  let config = {
    modelConfig: modelConfig,
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs/zh.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
