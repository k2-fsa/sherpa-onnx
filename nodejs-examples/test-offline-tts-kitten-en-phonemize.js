// Copyright (c)  2026  Xiaomi Corporation
//
// The text is phonemized outside of sherpa-onnx with piper-phonemize
// (https://www.npmjs.com/package/piper-phonemize) and the resulting phoneme
// codepoints are passed via GenerationConfig.phonemeCodepoints. No lexicon
// file and no espeak-ng-data are needed. If you want to use a lexicon
// instead, please see test-offline-tts-kitten-en-lexicon.js.

const sherpa_onnx = require('sherpa-onnx');
const piperPhonemize = require('piper-phonemize');

function createOfflineTts() {
  let offlineTtsKittenModelConfig = {
    model: './kitten-nano-en-v0_1-fp16/model.fp16.onnx',
    voices: './kitten-nano-en-v0_1-fp16/voices.bin',
    tokens: './kitten-nano-en-v0_1-fp16/tokens.txt',
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsKittenModelConfig: offlineTtsKittenModelConfig,
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
  sid: speakerId,
  speed: speed,
  silenceScale: 0.2,
};
// The text has already been phonemized; pass an empty string here.
const audio = tts.generateWithConfig('', generationConfig);
tts.save('./test-kitten-en-phonemize.wav', audio);
console.log('Saved to test-kitten-en-phonemize.wav successfully.');
tts.free();
