// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require("./index.js");

function createOfflineTts() {
  let vits = new sherpa_onnx.OfflineTtsVitsModelConfig();
  vits.model = "./vits-vctk/vits-vctk.onnx";
  vits.lexicon = "./vits-vctk/lexicon.txt";
  vits.tokens = "./vits-vctk/tokens.txt";

  let modelConfig = new sherpa_onnx.OfflineTtsModelConfig();
  modelConfig.vits = vits;

  let config = new sherpa_onnx.OfflineTtsConfig();
  config.model = modelConfig;

  return new sherpa_onnx.OfflineTts(config);
}

let tts = createOfflineTts();
let speakerId = 99;
let speed = 1.0;
let audio = tts.generate("Good morning. How are you doing?", speakerId, speed);
audio.save("./test-en.wav");
console.log("Saved to test-en.wav successfully.");
tts.free();
