// Copyright (c)  2026  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html
// to download model files
function createOfflineTts() {
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
      debug: true,
      numThreads: 2,
      provider: 'cpu',
    },
    maxNumSentences: 1,
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();

const text =
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.'

const referenceAudioFilename =
    './sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav';
const referenceWave = sherpa_onnx.readWave(referenceAudioFilename);

const generationConfig = new sherpa_onnx.GenerationConfig({
  speed: 1.0,
  referenceAudio: referenceWave.samples,
  referenceSampleRate: referenceWave.sampleRate,
  numSteps: 5,
  extra: {max_reference_audio_len: 12}
});


let start = Date.now();
const audio = tts.generate({text, generationConfig});

let stop = Date.now();
const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))

const filename = 'test-pocket-bria.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
