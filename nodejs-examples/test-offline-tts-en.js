// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  const vits = new sherpa_onnx.OfflineTtsVitsModelConfig();
  vits.model = './vits-vctk/vits-vctk.onnx';
  vits.lexicon = './vits-vctk/lexicon.txt';
  vits.tokens = './vits-vctk/tokens.txt';

  const modelConfig = new sherpa_onnx.OfflineTtsModelConfig();
  modelConfig.vits = vits;

  const config = new sherpa_onnx.OfflineTtsConfig();
  config.model = modelConfig;

  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();
const speakerId = 99;
const speed = 1.0;
const audio =
    tts.generate('Good morning. How are you doing?', speakerId, speed);
audio.save('./test-en.wav');
console.log('Saved to test-en.wav successfully.');
tts.free();
