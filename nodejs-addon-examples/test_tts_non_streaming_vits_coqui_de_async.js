// Copyright (c)  2026  Xiaomi Corporation
//
// Asynchronous text-to-speech with the VITS Coqui German model.
//
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
  const config = {
    model: {
      vits: {
        model: './vits-coqui-de-css10/model.onnx',
        tokens: './vits-coqui-de-css10/tokens.txt',
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

  const text = 'Alles hat ein Ende, nur die Wurst hat zwei.';

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

  const filename = 'test-vits-coqui-de-async.wav';
  sherpa_onnx.writeWave(
      filename, {samples: audio.samples, sampleRate: audio.sampleRate});
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exitCode = 1;
});
