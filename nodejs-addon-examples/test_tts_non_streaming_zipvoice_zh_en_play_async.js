// Copyright (c)  2026  Xiaomi Corporation
//
// npm install speaker
//
const Speaker = require('speaker');
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

function createSpeaker(sampleRate) {
  return new Speaker({
    channels: 1,
    bitDepth: 16,
    sampleRate: sampleRate,
    signed: true,
  });
}

function float32ToInt16Buffer(samples) {
  const buffer = Buffer.alloc(samples.length * 2);

  for (let i = 0; i < samples.length; ++i) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    const v = s < 0 ? s * 0x8000 : s * 0x7fff;
    buffer.writeInt16LE(Math.round(v), i * 2);
  }

  return buffer;
}

function waitForEvent(emitter, eventName) {
  return new Promise((resolve, reject) => {
    emitter.once(eventName, resolve);
    emitter.once('error', reject);
  });
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

  const speaker = createSpeaker(tts.sampleRate);
  const start = Date.now();

  console.log('Starting generation and playback...');

  const audio = await tts.generateAsync({
    text,
    enableExternalBuffer: true,
    generationConfig,
    onProgress: ({samples, progress}) => {
      process.stdout.write(
          `Progress: ${(progress * 100).toFixed(1)}%, ` +
          `Chunk samples: ${samples.length}\r`);
      speaker.write(float32ToInt16Buffer(samples));
      return 1;
    },
  });

  const generationStop = Date.now();
  speaker.end();
  await waitForEvent(speaker, 'close');
  const playbackStop = Date.now();

  console.log('\nGeneration and playback complete!');
  return {
    audio,
    generationElapsedSeconds: (generationStop - start) / 1000,
    playbackElapsedSeconds: (playbackStop - start) / 1000,
  };
}

async function main() {
  console.log('Creating OfflineTts...');
  const tts = await createOfflineTts();
  console.log('OfflineTts created!');

  const text =
      '小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中.';

  const {audio, generationElapsedSeconds, playbackElapsedSeconds} =
      await generateAudioAsync(tts, text);
  const duration = audio.samples.length / audio.sampleRate;
  const real_time_factor = generationElapsedSeconds / duration;

  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log(
      'Generation elapsed', generationElapsedSeconds.toFixed(3), 'seconds');
  console.log(
      'Playback drained in', playbackElapsedSeconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${generationElapsedSeconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));

  const filename = 'test-zipvoice-zh-en-play-async.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
});
