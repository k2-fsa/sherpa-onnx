// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require("./index.js");

function createOfflineTts() {
  let vits = new sherpa_onnx.OfflineTtsVitsModelConfig();
  vits.model = "./vits-zh-aishell3/vits-aishell3.onnx";
  vits.lexicon = "./vits-zh-aishell3/lexicon.txt";
  vits.tokens = "./vits-zh-aishell3/tokens.txt";

  let modelConfig = new sherpa_onnx.OfflineTtsModelConfig();
  modelConfig.vits = vits;

  let config = new sherpa_onnx.OfflineTtsConfig();
  config.model = modelConfig;
  config.ruleFsts = "./vits-zh-aishell3/rule.fst";

  return new sherpa_onnx.OfflineTts(config);
}

let tts = createOfflineTts();
let speakerId = 66;
let speed = 1.0;
let audio = tts.generate("3年前中国总人口是1411778724人", speakerId, speed);
audio.save("./test-zh.wav");
console.log("Saved to test-zh.wav successfully.");
tts.free();
