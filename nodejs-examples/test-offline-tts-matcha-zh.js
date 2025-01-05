// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx');

function createOfflineTts() {
  let offlineTtsMatchaModelConfig = {
    acousticModel: './matcha-icefall-zh-baker/model-steps-3.onnx',
    vocoder: './hifigan_v2.onnx',
    lexicon: './matcha-icefall-zh-baker/lexicon.txt',
    tokens: './matcha-icefall-zh-baker/tokens.txt',
    dictDir: './matcha-icefall-zh-baker/dict',
    noiseScale: 0.667,
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsMatchaModelConfig: offlineTtsMatchaModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };

  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    maxNumSentences: 1,
    ruleFsts:
        './matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst',
  };

  return sherpa_onnx.createOfflineTts(offlineTtsConfig);
}

const tts = createOfflineTts();
const speakerId = 0;
const speed = 1.0;
const text =
    '当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔. 某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。'

const audio = tts.generate({text: text, sid: speakerId, speed: speed});
tts.save('./test-matcha-zh.wav', audio);
console.log('Saved to test-matcha-zh.wav successfully.');
tts.free();
