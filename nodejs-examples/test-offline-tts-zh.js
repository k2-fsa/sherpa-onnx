// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  let offlineTtsVitsModelConfig = {
    model: './vits-zh-aishell3/vits-aishell3.onnx',
    lexicon: './vits-zh-aishell3/lexicon.txt',
    tokens: './vits-zh-aishell3/tokens.txt',
    dataDir: '',
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };

  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    ruleFsts: './vits-zh-aishell3/rule.fst',
    maxNumSentences: 1,
  };

  return sherpa_onnx.createOfflineTts(offlineTtsConfig);
}


const tts = createOfflineTts();
const speakerId = 66;
const speed = 1.0;
const audio = tts.generate(
    {text: '3年前中国总人口是1411778724人', sid: speakerId, speed: speed});
tts.save('./test-zh.wav', audio);
console.log('Saved to test-zh.wav successfully.');
tts.free();
