// Copyright (c)  2026  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
// to download model files
function createOfflineTts() {
  const config = {
    model: {
      supertonic: {
        durationPredictor:
            './sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx',
        textEncoder:
            './sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx',
        vectorEstimator:
            './sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx',
        vocoder:
            './sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx',
        ttsJson: './sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json',
        unicodeIndexer:
            './sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin',
        voiceStyle: './sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin',
      },
      debug: true,
      numThreads: 2,
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
  speed: 1.25,
  numSteps: 5,
  extra: {lang: 'en'},
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

const filename = 'test-supertonic-en.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
