// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  const vits = new sherpa_onnx.OfflineTtsVitsModelConfig();
  vits.model = 'vits-piper-en_US-amy-low/en_US-amy-low.onnx'
  vits.tokens = './vits-piper-en_US-amy-low/tokens.txt';
  vits.dataDir = './vits-piper-en_US-amy-low/espeak-ng-data'

  const modelConfig = new sherpa_onnx.OfflineTtsModelConfig();
  modelConfig.vits = vits;

  const config = new sherpa_onnx.OfflineTtsConfig();
  config.model = modelConfig;

  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();
const speakerId = 0;
const speed = 1.0;
const audio = tts.generate(
    '“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.”',
    speakerId, speed);
audio.save('./test-en.wav');
console.log('Saved to test-en.wav successfully.');
tts.free();
