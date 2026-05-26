// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

// please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
// to download model files
function createOfflineTts() {
  let supertonic = {
    durationPredictor:
        './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx',
    textEncoder:
        './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx',
    vectorEstimator:
        './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx',
    vocoder: './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx',
    ttsJson: './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json',
    unicodeIndexer:
        './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin',
    voiceStyle: './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin',
  };
  let offlineTtsModelConfig = {
    offlineTtsSupertonicModelConfig: supertonic,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };

  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    maxNumSentences: 1,
  };

  return sherpa_onnx.createOfflineTts(offlineTtsConfig);
}

const tts = createOfflineTts();
const speakerId = 0;  // it has 10 speakers, valid speakerId [0,9]
const speed = 1.0;
const numSteps = 8;

// SupertonicTTS supports 31 languages. Use --lang to select one.
// Supported language codes:
//   ar, bg, cs, da, de, el, en, es, et, fi, fr, hi, hr, hu,
//   id, it, ja, ko, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv,
//   tr, uk, vi
//
// See also https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html

const samples = [
  {
    lang: 'en',
    text:
        'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.',
  },
  {
    lang: 'ja',
    text: 'これは次世代のkaldiを使用したテキスト読み上げエンジンです。',
  },
  {
    lang: 'de',
    text: 'Alles hat ein Ende, nur die Wurst hat zwei.',
  },
];

for (const s of samples) {
  const generationConfig = {
    sid: speakerId,
    speed: speed,
    numSteps: numSteps,
    extra: {lang: s.lang},
  };

  const audio = tts.generateWithConfig(s.text, generationConfig);
  const filename = `./test-supertonic-${s.lang}.wav`;
  tts.save(filename, audio);
  console.log(`Saved to ${filename} successfully.`);
}

tts.free();
