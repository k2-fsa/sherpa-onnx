// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const portAudio = require("naudiodon2");
// console.log(portAudio.getDevices());

const sherpa_onnx = require("./index.js");

function createRecognizer() {
  let featConfig = new sherpa_onnx.FeatureConfig();
  featConfig.sampleRate = 16000;
  featConfig.featureDim = 80;

  // test online recognizer
  let transducer = new sherpa_onnx.OnlineTransducerModelConfig();
  transducer.encoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx";
  transducer.decoder =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx";
  transducer.joiner =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx";
  let tokens =
      "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt";

  let modelConfig = new sherpa_onnx.OnlineModelConfig();
  modelConfig.transducer = transducer;
  modelConfig.tokens = tokens;
  modelConfig.modelType = "zipformer";

  let recognizerConfig = new sherpa_onnx.OnlineRecognizerConfig();
  recognizerConfig.featConfig = featConfig;
  recognizerConfig.modelConfig = modelConfig;
  recognizerConfig.decodingMethod = "greedy_search";
  recognizerConfig.enableEndpoint = 1;

  let recognizer = new sherpa_onnx.OnlineRecognizer(recognizerConfig);
  return recognizer;
}
recognizer = createRecognizer();
stream = recognizer.createStream();
display = new sherpa_onnx.Display(50);

let lastText = "";
let segmentIndex = 0;

let ai = new portAudio.AudioIO({
  inOptions : {
    channelCount : 1,
    closeOnError : true, // Close the stream if an audio error is detected, if
                         // set false then just log the error
    deviceId : -1, // Use -1 or omit the deviceId to select the default device
    sampleFormat : portAudio.SampleFormatFloat32,
    sampleRate : recognizer.config.featConfig.sampleRate
  }
});

ai.on("data", data => {
  let samples = new Float32Array(data.buffer);

  stream.acceptWaveform(recognizer.config.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  let isEndpoint = recognizer.isEndpoint(stream);
  let text = recognizer.getResult(stream).text;

  if (text.length > 0 && lastText != text) {
    lastText = text;
    display.print(segmentIndex, lastText);
  }
  if (isEndpoint) {
    if (text.length > 0) {
      lastText = text;
      segmentIndex += 1;
    }
    recognizer.reset(stream)
  }
});

ai.on("close", () => {
  console.log("Free resources");
  stream.free();
  recognizer.free();
});

ai.start();
console.log("Started! Please speak")
