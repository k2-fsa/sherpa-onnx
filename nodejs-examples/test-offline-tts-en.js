// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  let offlineTtsVitsModelConfig = {
    model: './vits-piper-en_US-amy-low/en_US-amy-low.onnx',
    lexicon: '',
    tokens: './vits-piper-en_US-amy-low/tokens.txt',
    dataDir: './vits-piper-en_US-amy-low/espeak-ng-data',
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };

  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    ruleFsts: '',
    ruleFars: '',
    maxNumSentences: 1,
  };

  return sherpa_onnx.createOfflineTts(offlineTtsConfig);
}


const tts = createOfflineTts();
const speakerId = 0;
const speed = 1.0;
const audio = tts.generate({
  text:
      '“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.”',
  sid: speakerId,
  speed: speed
});

tts.save('./test-en.wav', audio);
console.log('Saved to test-en.wav successfully.');

tts.free();
