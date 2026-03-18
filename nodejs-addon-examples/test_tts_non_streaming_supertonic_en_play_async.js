// Copyright (c)  2026  Xiaomi Corporation
//
// npm install speaker
//
const Speaker = require('speaker');
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
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
  const generationConfig = new sherpa_onnx.GenerationConfig({
    sid: 6,
    speed: 1.25,
    numSteps: 5,
    extra: {lang: 'en'},
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
      'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

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

  const filename = 'test-supertonic-en-play-async.wav';
  sherpa_onnx.writeWave(filename, {
    samples: audio.samples,
    sampleRate: audio.sampleRate,
  });
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
});
