// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  let offlineTtsKokoroModelConfig = {
    model: './kokoro-en-v0_19/model.onnx',
    voices: './kokoro-en-v0_19/voices.bin',
    tokens: './kokoro-en-v0_19/tokens.txt',
    dataDir: './kokoro-en-v0_19/espeak-ng-data',
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsKokoroModelConfig: offlineTtsKokoroModelConfig,
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
tts.save('./test-kokoro-en.wav', audio);
console.log('Saved to test-kokoro-en.wav successfully.');
tts.free();
