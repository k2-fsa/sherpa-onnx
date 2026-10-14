// Copyright (c)  2026  Xiaomi Corporation
//
// The text is phonemized outside of sherpa-onnx with piper-phonemize
// (https://www.npmjs.com/package/piper-phonemize-node) and the resulting
// phoneme codepoints are passed via GenerationConfig.phonemeCodepoints. No
// lexicon file and no espeak-ng-data are needed. If you want to use a lexicon
//
//
// There is no German lexicon, so this model is phonemize-only.
const sherpa_onnx = require('sherpa-onnx-node');
const piperPhonemize = require('piper-phonemize-node');

// please download model files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
function createOfflineTts() {
  const config = {
    model: {
      vits: {
        model: './vits-piper-de_DE-glados-low/de_DE-glados-low.onnx',
        tokens: './vits-piper-de_DE-glados-low/tokens.txt',
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

const text = 'Alles hat ein Ende, nur die Wurst hat zwei.';

// The espeak-ng voice is taken from the model metadata. Fall back to
// de if the model has no language information.
const voice = tts.lang || 'de';
const phonemeCodepoints = piperPhonemize.phonemize(text, voice)
    // piper may emit empty sentences (e.g. a trailing segment after
    // punctuation); the C API requires every sentence to be non-empty.
    .filter((sentence) => sentence.length > 0);

const generationConfig = new sherpa_onnx.GenerationConfig({
  phonemeCodepoints,
  sid: 0,
  speed: 1.0,
  silenceScale: 0.2,
});


let start = Date.now();
// The text has already been phonemized; pass an empty string here.
const audio = tts.generate({text: '', generationConfig});
let stop = Date.now();
const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds');
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3));

const filename = 'test-vits-de-phonemize.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
