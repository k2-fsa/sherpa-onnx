// Copyright (c)  2025  Xiaomi Corporation

const sherpa_onnx = require('sherpa-onnx-node');

/**
 * Create an offline TTS instance using the Kitten model.
 *
 * Model files can be downloaded from:
 * https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html
 */
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

async function main() {
  const tts = createOfflineTts();

  const text =
      'Today as always, men fall into two groups: slaves and free men.' +
      'Whoever does not have two-thirds of his day for himself, is a slave,' +
      'whatever he may be: a statesman, a businessman, an official, or a scholar.';

  console.log('Number of speakers:', tts.numSpeakers);
  console.log('Sample rate:', tts.sampleRate);

  const start = Date.now();

  // Asynchronous generation with progress reporting
  const audio = await tts.generateAsync({
    text,
    sid: 6,
    speed: 1.0,

    // Progress callback receives audio chunks
    onProgress({samples, progress}) {
      // samples is Float32Array for this chunk
      process.stdout.write(`\rGenerating... ${
          (progress * 100).toFixed(1)}% (chunk length: ${samples.length})`);

      // Return 0 or false to cancel, any other value to continue
      return false;
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

  const filename = 'test-kitten-en-6.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });

  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('TTS failed:', err);
  process.exit(1);
});
