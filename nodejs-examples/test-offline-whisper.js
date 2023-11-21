// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require("fs");
const {Readable} = require("stream");
const wav = require("wav");

const sherpa_onnx = require("./index.js");

function createRecognizer() {
  let featConfig = new sherpa_onnx.FeatureConfig();
  featConfig.sampleRate = 16000;
  featConfig.featureDim = 80;

  // test online recognizer
  let whisper = new sherpa_onnx.OfflineWhisperModelConfig();
  whisper.encoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx";
  whisper.decoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx";
  let tokens = "./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt";

  let modelConfig = new sherpa_onnx.OfflineModelConfig();
  modelConfig.whisper = whisper;
  modelConfig.tokens = tokens;
  modelConfig.modelType = "whisper";

  let recognizerConfig = new sherpa_onnx.OfflineRecognizerConfig();
  recognizerConfig.featConfig = featConfig;
  recognizerConfig.modelConfig = modelConfig;
  recognizerConfig.decodingMethod = "greedy_search";

  let recognizer = new sherpa_onnx.OfflineRecognizer(recognizerConfig);
  return recognizer;
}

recognizer = createRecognizer();
stream = recognizer.createStream();

const waveFilename = "./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav";

const reader = new wav.Reader();
const readable = new Readable().wrap(reader);
let buf = [];

reader.on("format", ({audioFormat, bitDepth, channels, sampleRate}) => {
  if (sampleRate != recognizer.config.featConfig.sampleRate) {
    throw new Error(`Only support sampleRate ${
        recognizer.config.featConfig.sampleRate}. Given ${sampleRate}`);
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

fs.createReadStream(waveFilename, {"highWaterMark" : 4096})
    .pipe(reader)
    .on("finish", function(err) {
      // tail padding
      const floatSamples =
          new Float32Array(recognizer.config.featConfig.sampleRate * 0.5);

      buf.push(floatSamples);
      let flattened = Float32Array.from(buf.reduce((a, b) => [...a, ...b], []));

      stream.acceptWaveform(recognizer.config.featConfig.sampleRate, flattened);
      recognizer.decode(stream);
      const r = recognizer.getResult(stream);
      console.log(r.text);

      stream.free();
      recognizer.free();
    });

readable.on("readable", function() {
  let chunk;
  while ((chunk = readable.read()) != null) {
    const int16Samples =
        new Int16Array(chunk.buffer, chunk.byteOffset,
                       chunk.length / Int16Array.BYTES_PER_ELEMENT);

    let floatSamples = new Float32Array(int16Samples.length);

    for (let i = 0; i < floatSamples.length; i++) {
      floatSamples[i] = int16Samples[i] / 32768.0;
    }

    buf.push(floatSamples);
  }
});
