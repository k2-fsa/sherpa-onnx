// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createRecognizer() {
  const featConfig = new sherpa_onnx.FeatureConfig();
  featConfig.sampleRate = 16000;
  featConfig.featureDim = 80;

  // test online recognizer
  const zipformer2Ctc = new sherpa_onnx.OnlineZipformer2CtcModelConfig();
  zipformer2Ctc.model =
      './sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx';
  const tokens =
      './sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt';

  const modelConfig = new sherpa_onnx.OnlineModelConfig();
  modelConfig.zipformer2Ctc = zipformer2Ctc;
  modelConfig.tokens = tokens;

  const recognizerConfig = new sherpa_onnx.OnlineRecognizerConfig();
  recognizerConfig.featConfig = featConfig;
  recognizerConfig.modelConfig = modelConfig;
  recognizerConfig.decodingMethod = 'greedy_search';

  recognizer = new sherpa_onnx.OnlineRecognizer(recognizerConfig);
  return recognizer;
}
recognizer = createRecognizer();
stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav';

const reader = new wav.Reader();
const readable = new Readable().wrap(reader);

function decode(samples) {
  stream.acceptWaveform(recognizer.config.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }
  const r = recognizer.getResult(stream);
  console.log(r.text);
}

reader.on('format', ({audioFormat, bitDepth, channels, sampleRate}) => {
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

fs.createReadStream(waveFilename, {'highWaterMark': 4096})
    .pipe(reader)
    .on('finish', function(err) {
      // tail padding
      const floatSamples =
          new Float32Array(recognizer.config.featConfig.sampleRate * 0.5);
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

    const floatSamples = new Float32Array(int16Samples.length);

    for (let i = 0; i < floatSamples.length; i++) {
      floatSamples[i] = int16Samples[i] / 32768.0;
    }

    decode(floatSamples);
  }
});
