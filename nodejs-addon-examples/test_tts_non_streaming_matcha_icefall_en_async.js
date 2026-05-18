// Copyright (c)  2026  Xiaomi Corporation
//
// Asynchronous text-to-speech with the Matcha English model.
//
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
  const config = {
    model: {
      matcha: {
        acousticModel: './matcha-icefall-en_US-ljspeech/model-steps-3.onnx',
        vocoder: './vocos-22khz-univ.onnx',
        tokens: './matcha-icefall-en_US-ljspeech/tokens.txt',
        dataDir: './matcha-icefall-en_US-ljspeech/espeak-ng-data',
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

  const generationConfig = new sherpa_onnx.GenerationConfig({
    sid: 0,
    speed: 1.0,
    silenceScale: 0.2,
  });

  const start = Date.now();
  const audio = await tts.generateAsync({
    text,
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

  const filename = 'test-matcha-en-async.wav';
  sherpa_onnx.writeWave(
      filename, {samples: audio.samples, sampleRate: audio.sampleRate});
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exitCode = 1;
});
