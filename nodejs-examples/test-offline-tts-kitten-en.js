// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  let offlineTtsKittenModelConfig = {
    model: './kitten-nano-en-v0_1-fp16/model.fp16.onnx',
    voices: './kitten-nano-en-v0_1-fp16/voices.bin',
    tokens: './kitten-nano-en-v0_1-fp16/tokens.txt',
    dataDir: './kitten-nano-en-v0_1-fp16/espeak-ng-data',
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
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.'

const audio = tts.generate({text: text, sid: speakerId, speed: speed});
tts.save('./test-kitten-en.wav', audio);
console.log('Saved to test-kitten-en.wav successfully.');
tts.free();
