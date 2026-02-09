// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  let pocket = {
    lmFlow: './sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx',
    lmMain: './sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx',
    encoder: './sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx',
    decoder: './sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx',
    textConditioner:
        './sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx',
    vocabJson: './sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json',
    tokenScoresJson:
        './sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json',
  };
  let offlineTtsModelConfig = {
    offlineTtsPocketModelConfig: pocket,
    numThreads: 1,
    debug: 1,  // set it to 1 to see verbose logs; 0 to disable logs
    provider: 'cpu',
  };

  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    maxNumSentences: 1,
  };

  return sherpa_onnx.createOfflineTts(offlineTtsConfig);
}

const referenceWaveFilename =
    './sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav';
const wave = sherpa_onnx.readWave(referenceWaveFilename);

const generationConfig = {
  silenceScale: 0.2,
  referenceAudio: wave.samples,
  referenceSampleRate: wave.sampleRate,
  numSteps: 5,
  extra: {max_reference_audio_len: 12}
};

const tts = createOfflineTts();
const text =
    'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

const audio = tts.generateWithConfig(text, generationConfig);
tts.save('./test-pocket-en.wav', audio);
console.log('Saved to test-pocket-en.wav successfully.');
tts.free();
