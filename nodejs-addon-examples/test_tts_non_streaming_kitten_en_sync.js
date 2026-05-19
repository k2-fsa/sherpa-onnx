// Copyright (c)  2026  Xiaomi Corporation
//
// Synchronous text-to-speech with the Kitten model.
//
const sherpa_onnx = require('sherpa-onnx-node');

function createOfflineTts() {
  const config = {
    model: {
      kitten: {
        model: './kitten-nano-en-v0_1-fp16/model.fp16.onnx',
        voices: './kitten-nano-en-v0_1-fp16/voices.bin',
        tokens: './kitten-nano-en-v0_1-fp16/tokens.txt',
        dataDir: './kitten-nano-en-v0_1-fp16/espeak-ng-data',
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
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

const generationConfig = new sherpa_onnx.GenerationConfig({
  sid: 6,
  speed: 1.0,
  silenceScale: 0.2,
});

let start = Date.now();
const audio = tts.generate({text, generationConfig});
let stop = Date.now();
const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds');
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3));

const filename = 'test-kitten-en-sync.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
