// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');
var portAudio = require('naudiodon2');
console.log(portAudio.getDevices());


const sherpa_onnx = require('./index.js');

let featConfig = new sherpa_onnx.FeatureConfig()
featConfig.sampleRate = 16000;
featConfig.featureDim = 80;

// test online recognizer
let transducer = new sherpa_onnx.OnlineTransducerModelConfig();
transducer.encoder =
    './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx'
transducer.decoder =
    './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.int8.onnx'
transducer.joiner =
    './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx'
let tokens =
    './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt'

let modelConfig = new sherpa_onnx.OnlineModelConfig()
modelConfig.transducer = transducer;
modelConfig.tokens = tokens;
modelConfig.debug = 1;
modelConfig.modelType = 'zipformer';

let recognizerConfig = new sherpa_onnx.OnlineRecognizerConfig()
recognizerConfig.featConfig = featConfig;
recognizerConfig.modelConfig = modelConfig;
recognizerConfig.decodingMethod = 'greedy_search';

recognizer = new sherpa_onnx.OnlineRecognizer(recognizerConfig);
stream = recognizer.createStream()

var ai = new portAudio.AudioIO({
  inOptions: {
    channelCount: 1,
    sampleFormat: portAudio.SampleFormatFloat32,
    sampleRate: featConfig.sampleRate,
    deviceId: -1,  // Use -1 or omit the deviceId to select the default device
    closeOnError: true  // Close the stream if an audio error is detected, if
                        // set false then just log the error
  }
});

ai.on('data', buf => {
  let samples = new Float32Array(buf.buffer);
  stream.acceptWaveform(recognizerConfig.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }
  const r = recognizer.getResult(stream);
  console.log(r.text);
});


// Start streaming
ai.start();
