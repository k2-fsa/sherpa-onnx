// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('./index.js');

let featConfig = new sherpa_onnx.FeatureConfig()
featConfig.sampleRate = 16000;
featConfig.featureDim = 80;

let paraformer = new sherpa_onnx.OnlineParaformerModelConfig();
paraformer.encoder =
    './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx'
paraformer.decoder =
    './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx'
let tokens = './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt'

let modelConfig = new sherpa_onnx.OnlineModelConfig()
modelConfig.paraformer = paraformer;
modelConfig.tokens = tokens;
modelConfig.debug = 1;
modelConfig.modelType = 'paraformer';

let recognizerConfig = new sherpa_onnx.OnlineRecognizerConfig()
recognizerConfig.featConfig = featConfig;
recognizerConfig.modelConfig = modelConfig;
recognizerConfig.decodingMethod = 'greedy_search';

console.log(recognizerConfig);

recognizer = new sherpa_onnx.OnlineRecognizer(recognizerConfig);
stream = recognizer.createStream()

const waveFilename =
    './sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/0.wav'

const reader = new wav.Reader();
const readable = new Readable().wrap(reader);

function decode(samples) {
  stream.acceptWaveform(recognizerConfig.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }
  const r = recognizer.getResult(stream);
  console.log(r.text);
}

reader.on('format', ({audioFormat, sampleRate, channels, bitDepth}) => {
  if (sampleRate != featConfig.sampleRate) {
    throw new Error(`Only support sampleRate ${featConfig.sampleRate}. Given ${
        sampleRate}`);
  }

  if (audioFormat != 1) {
    throw new Error(`Only support PCM format. Given ${audioFormat}`);
  }

  if (channels != 1) {
    throw new Error(`Only a single channel. Given ${channel}`);
  }

  if (bitDepth != 16) {
    throw new Error(`Only support 16-bit samples. Given ${bitDepth}`);
  }
});

fs.createReadStream(waveFilename, {'highWaterMark': 4096})
    .pipe(reader)
    .on('finish', function(err) {
      // tail padding
      const floatSamples =
          new Float32Array(recognizerConfig.featConfig.sampleRate * 0.5);
      decode(floatSamples);
      stream.free();
      recognizer.free();
    });


readable.on('readable', function() {
  let chunk;
  while ((chunk = readable.read()) != null) {
    const int16Samples = new Int16Array(
        chunk.buffer, chunk.byteOffset,
        chunk.length / Int16Array.BYTES_PER_ELEMENT);

    let floatSamples = new Float32Array(int16Samples.length);
    for (let i = 0; i < floatSamples.length; i++) {
      floatSamples[i] = int16Samples[i] / 32768.0;
    }
    decode(floatSamples);
  }
});
