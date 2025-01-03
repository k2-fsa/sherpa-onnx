// Copyright (c)  2025  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker
// to download model files
function createOfflineTts() {
  const config = {
    model: {
      matcha: {
        acousticModel: './matcha-icefall-en_US-ljspeech/model-steps-3.onnx',
        vocoder: './hifigan_v2.onnx',
        lexicon: './matcha-icefall-en_US-ljspeech/lexicon.txt',
        tokens: './matcha-icefall-en_US-ljspeech/tokens.txt',
        dataDir: './matcha-icefall-en_US-ljspeech/espeak-ng-data',
      },
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();

const text =
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.'


let start = Date.now();
const audio = tts.generate({text: text, sid: 0, speed: 1.0});
let stop = Date.now();
const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'secodns')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'secodns')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))

const filename = 'test-matcha-en.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
