// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  let offlineTtsMatchaModelConfig = {
    acousticModel: './matcha-icefall-en_US-ljspeech/model-steps-3.onnx',
    vocoder: './hifigan_v2.onnx',
    lexicon: './matcha-icefall-en_US-ljspeech/lexicon.txt',
    tokens: './matcha-icefall-en_US-ljspeech/tokens.txt',
    dataDir: './matcha-icefall-en_US-ljspeech/espeak-ng-data',

    noiseScale: 0.667,
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsMatchaModelConfig: offlineTtsMatchaModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };

  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    maxNumSentences: 1,
  };

  return sherpa_onnx.createOfflineTts(offlineTtsConfig);
}

const tts = createOfflineTts();
const speakerId = 0;
const speed = 1.0;
const text =
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.'

const audio = tts.generate({text: text, sid: speakerId, speed: speed});
tts.save('./test-matcha-en.wav', audio);
console.log('Saved to test-matcha-en.wav successfully.');
tts.free();
