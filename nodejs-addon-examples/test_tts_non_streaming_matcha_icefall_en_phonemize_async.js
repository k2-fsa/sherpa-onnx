// Copyright (c)  2026  Xiaomi Corporation
//
// The text is phonemized outside of sherpa-onnx with piper-phonemize
// (https://www.npmjs.com/package/piper-phonemize-node) and the resulting
// phoneme codepoints are passed via GenerationConfig.phonemeCodepoints. No
// lexicon file and no espeak-ng-data are needed. If you want to use a lexicon
// instead, please see test_tts_non_streaming_matcha_icefall_en_lexicon_async.js.
//
// Asynchronous text-to-speech with the Matcha English model.
//
const sherpa_onnx = require('sherpa-onnx-node');
const piperPhonemize = require('piper-phonemize-node');

async function createOfflineTts() {
  const config = {
    model: {
      matcha: {
        acousticModel: './matcha-icefall-en_US-ljspeech/model-steps-3.onnx',
        vocoder: './vocos-22khz-univ.onnx',
        tokens: './matcha-icefall-en_US-ljspeech/tokens.txt',
      },
      debug: false,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };
  return await sherpa_onnx.OfflineTts.createAsync(config);
}

async function main() {
  const tts = await createOfflineTts();

  const text =
      'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

  // The espeak-ng voice is taken from the model metadata. Fall back to
  // en-us if the model has no language information.
  const voice = tts.lang || 'en-us';
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

  const start = Date.now();
  // The text has already been phonemized; pass an empty string here.
  const audio = await tts.generateAsync({
    text: '',
    enableExternalBuffer: true,
    generationConfig,
    onProgress: ({samples, progress}) => {
      process.stdout.write(
          `Progress: ${(progress * 100).toFixed(1)}%, ` +
          `Samples: ${samples.length}\r`);
      return 1;
    },
  });

  console.log('');
  const stop = Date.now();
  const elapsed_seconds = (stop - start) / 1000;
  const duration = audio.samples.length / audio.sampleRate;
  const real_time_factor = elapsed_seconds / duration;
  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));

  const filename = 'test-matcha-en-phonemize-async.wav';
  sherpa_onnx.writeWave(
      filename, {samples: audio.samples, sampleRate: audio.sampleRate});
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exitCode = 1;
});
