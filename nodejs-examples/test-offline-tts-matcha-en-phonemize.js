// Copyright (c)  2026  Xiaomi Corporation
//
// The text is phonemized outside of sherpa-onnx with piper-phonemize
// (https://www.npmjs.com/package/piper-phonemize) and the resulting phoneme
// codepoints are passed via GenerationConfig.phonemeCodepoints. No lexicon
// file and no espeak-ng-data are needed. If you want to use a lexicon
// instead, please see test-offline-tts-matcha-en-lexicon.js.

const sherpa_onnx = require('sherpa-onnx');
const piperPhonemize = require('piper-phonemize');

const silenceScale = 0.2;

function createOfflineTts() {
  let offlineTtsMatchaModelConfig = {
    acousticModel: './matcha-icefall-en_US-ljspeech/model-steps-3.onnx',
    vocoder: './vocos-22khz-univ.onnx',
    tokens: './matcha-icefall-en_US-ljspeech/tokens.txt',

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
    silenceScale: silenceScale,
  };

  return sherpa_onnx.createOfflineTts(offlineTtsConfig);
}

const tts = createOfflineTts();
const text =
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

// The espeak-ng voice is taken from the model metadata. Fall back to
// en-us if the model has no language information.
const voice = tts.lang || 'en-us';
const phonemeCodepoints = piperPhonemize.phonemize(text, voice)
    // piper may emit empty sentences (e.g. a trailing segment after
    // punctuation); the C API requires every sentence to be non-empty.
    .filter((sentence) => sentence.length > 0);

const generationConfig = {
  phonemeCodepoints: phonemeCodepoints,
  sid: 0,
  speed: 1.0,
  silenceScale: silenceScale,
};

// The text has already been phonemized; pass an empty string here.
const audio = tts.generateWithConfig('', generationConfig);
tts.save('./test-matcha-en-phonemize.wav', audio);
console.log('Saved to test-matcha-en-phonemize.wav successfully.');
tts.free();
