// Copyright (c)  2026  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');
const fs = require('fs');

async function createOfflineTts() {
  const config = {
    model: {
      pocket: {
        lmFlow: './sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx',
        lmMain: './sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx',
        encoder: './sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx',
        decoder: './sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx',
        textConditioner:
            './sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx',
        vocabJson: './sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json',
        tokenScoresJson:
            './sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json',
      },
      debug: false,  // set to true to see verbose logs
      numThreads: 2,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };

  return await sherpa_onnx.OfflineTts.createAsync(config);
}

/**
 * Async function to generate audio with progress callback
 * @param {sherpa_onnx.OfflineTts} tts
 * @param {string} text
 */
async function generateAudioAsync(tts, text) {
  const referenceAudioFilename =
      './sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav';
  const referenceWave = sherpa_onnx.readWave(referenceAudioFilename);

  const generationConfig = new sherpa_onnx.GenerationConfig({
    speed: 1.0,
    referenceAudio: referenceWave.samples,
    referenceSampleRate: referenceWave.sampleRate,
    numSteps: 5,
    extra: {max_reference_audio_len: 12},
  });

  console.log('Starting generation...');

  const audio = await tts.generateAsyncWithConfig({
    text,
    enableExternalBuffer: true,
    generationConfig,
    onProgress: ({samples, progress}) => {
      // Print progress percentage and number of samples generated
      process.stdout.write(
          `Progress: ${(progress * 100).toFixed(1)}%, ` +
          `Samples: ${samples.length}\r`);

      // Return anything other than 0/false to continue generation
      return 1;
    },
  });

  console.log('\nGeneration complete!');
  return audio;
}

/**
 * Main entry
 */
async function main() {
  console.log('Creating OfflineTts...');
  const tts = await createOfflineTts();
  console.log('OfflineTts created!');

  const text =
      'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';


  const start = Date.now();
  const audio = await generateAudioAsync(tts, text);
  const stop = Date.now();

  const elapsed_seconds = (stop - start) / 1000;
  const duration = audio.samples.length / audio.sampleRate;
  const real_time_factor = elapsed_seconds / duration;

  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));

  const filename = 'test-pocket-bria-async.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
  console.log(`Saved to ${filename}`);
}

// Run the async main
main().catch((err) => {
  console.error('Error:', err);
});
