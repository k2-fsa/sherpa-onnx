// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createOnlineRecognizer() {
  let onlineTransducerModelConfig = {
    encoder:
        './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx',
    decoder:
        './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx',
    joiner:
        './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx',
  };

  let onlineParaformerModelConfig = {
    encoder: '',
    decoder: '',
  };

  let onlineZipformer2CtcModelConfig = {
    model: '',
  };

  let onlineModelConfig = {
    transducer: onlineTransducerModelConfig,
    paraformer: onlineParaformerModelConfig,
    zipformer2Ctc: onlineZipformer2CtcModelConfig,
    tokens:
        './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt',
    numThreads: 1,
    provider: 'cpu',
    debug: 1,
    modelType: 'zipformer',
  };

  let featureConfig = {
    sampleRate: 16000,
    featureDim: 80,
  };

  let recognizerConfig = {
    featConfig: featureConfig,
    modelConfig: onlineModelConfig,
    decodingMethod: 'greedy_search',
    maxActivePaths: 4,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 2.4,
    rule2MinTrailingSilence: 1.2,
    rule3MinUtteranceLength: 20,
    hotwordsFile: '',
    hotwordsScore: 1.5,
    ctcFstDecoderConfig: {
      graph: '',
      maxActive: 3000,
    }
  };

  return sherpa_onnx.createOnlineRecognizer(recognizerConfig);
}

const recognizer = createOnlineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav';

const reader = new wav.Reader();
const readable = new Readable().wrap(reader);

function decode(samples) {
  stream.acceptWaveform(recognizer.config.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }
  const text = recognizer.getResult(stream);
  console.log(text);
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
