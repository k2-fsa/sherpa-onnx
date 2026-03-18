// Copyright (c)  2026  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
  const config = {
    model: {
      zipvoice: {
        tokens: './sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt',
        encoder:
            './sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx',
        decoder:
            './sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx',
        vocoder: './vocos_24khz.onnx',
        dataDir:
            './sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data',
        lexicon: './sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt',
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
 * @param {sherpa_onnx.OfflineTts} tts
 * @param {string} text
 */
async function generateAudioAsync(tts, text) {
  const referenceText =
      '那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.';
  const referenceAudioFilename =
      './sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav';
  const referenceWave = sherpa_onnx.readWave(referenceAudioFilename);

  const generationConfig = new sherpa_onnx.GenerationConfig({
    speed: 1.0,
    referenceAudio: referenceWave.samples,
    referenceSampleRate: referenceWave.sampleRate,
    referenceText,
    numSteps: 4,
    extra: {min_char_in_sentence: 10},
  });

  console.log('Starting generation...');

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

  console.log('\nGeneration complete!');
  return audio;
}

async function main() {
  console.log('Creating OfflineTts...');
  const tts = await createOfflineTts();
  console.log('OfflineTts created!');

  const text =
      '小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中.';

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

  const filename = 'test-zipvoice-zh-en-async.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
});
