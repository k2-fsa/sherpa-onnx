// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');
const performance = require('perf_hooks').performance;

// please download model files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
function createOfflineTts() {
  const config = {
    model: {
      vits: {
        model: './vits-coqui-de-css10/model.onnx',
        tokens: './vits-coqui-de-css10/tokens.txt',
      },
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumStences: 1,
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();

const text = 'Alles hat ein Ende, nur die Wurst hat zwei.'

let start = performance.now();
const audio = tts.generate({text: text, sid: 0, speed: 1.0});
let stop = performance.now();
const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'secodns')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'secodns')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))

const filename = 'test-coqui-de.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
