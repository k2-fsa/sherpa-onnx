// Copyright (c)  2026  Xiaomi Corporation
//
// The text is phonemized outside of sherpa-onnx with piper-phonemize
// (https://www.npmjs.com/package/piper-phonemize-node) and the resulting
// phoneme codepoints are passed via GenerationConfig.phonemeCodepoints. No
// lexicon file and no espeak-ng-data are needed. If you want to use a lexicon
// instead, please see test_tts_non_streaming_kitten_en_lexicon.js.

const sherpa_onnx = require('sherpa-onnx-node');
const piperPhonemize = require('piper-phonemize-node');

/**
 * Create an offline TTS instance asynchronously using the Kitten model.
 *
 * Model files can be downloaded from:
 * https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html
 */
async function createOfflineTtsAsync() {
  const config = {
    model: {
      kitten: {
        model: './kitten-nano-en-v0_1-fp16/model.fp16.onnx',
        voices: './kitten-nano-en-v0_1-fp16/voices.bin',
        tokens: './kitten-nano-en-v0_1-fp16/tokens.txt',
      },
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };

  // Use the async factory (non-blocking)
  return await sherpa_onnx.OfflineTts.createAsync(config);
}

async function main() {
  // Asynchronously create the OfflineTts instance
  const tts = await createOfflineTtsAsync();

  const text =
      'Today as always, men fall into two groups: slaves and free men. ' +
      'Whoever does not have two-thirds of his day for himself, is a slave, ' +
      'whatever he may be: a statesman, a businessman, an official, or a scholar.';

  console.log('Number of speakers:', tts.numSpeakers);
  console.log('Sample rate:', tts.sampleRate);

  const start = Date.now();
  // The espeak-ng voice is taken from the model metadata. Fall back to
  // en-us if the model has no language information.
  const voice = tts.lang || 'en-us';
  const phonemeCodepoints = piperPhonemize.phonemize(text, voice)
    // piper may emit empty sentences (e.g. a trailing segment after
    // punctuation); the C API requires every sentence to be non-empty.
    .filter((sentence) => sentence.length > 0);

  const generationConfig = new sherpa_onnx.GenerationConfig({
    phonemeCodepoints,
    sid: 6,
    speed: 1.0,
    silenceScale: 0.2,
  });

  // Asynchronous generation with progress reporting
  // The text has already been phonemized; pass an empty string here.
  const audio = await tts.generateAsync({
    text: '',
    generationConfig,

    // Progress callback receives audio chunks
    onProgress({samples, progress}) {
      // samples is Float32Array for this chunk
      process.stdout.write(`\rGenerating... ${
          (progress * 100).toFixed(1)}% (chunk length: ${samples.length})`);

      // Return 0 or false to cancel, any other value to continue
      return true;
    },
  });

  console.log('\nGeneration finished.');

  const stop = Date.now();
  const elapsedSeconds = (stop - start) / 1000;
  const durationSeconds = audio.samples.length / audio.sampleRate;
  const realTimeFactor = elapsedSeconds / durationSeconds;

  console.log('Wave duration:', durationSeconds.toFixed(3), 'seconds');
  console.log('Elapsed time:', elapsedSeconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsedSeconds.toFixed(3)} / ${durationSeconds.toFixed(3)} =`,
      realTimeFactor.toFixed(3));

  const filename = 'test-kitten-en-6-phonemize.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });

  console.log(`Saved to ${filename}`);
}

// Run the demo
main().catch((err) => {
  console.error('TTS failed:', err);
  process.exit(1);
});
