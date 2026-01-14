// Copyright (c)  2026  Xiaomi Corporation
//  This file shows how to use the async API to decode multiple files
const path = require('path');
const sherpa_onnx = require('sherpa-onnx-node');

/**
 * Create an OfflineRecognizer with FunASR Nano model.
 */
function createRecognizer(modelDir, numThreads = 2, debug = 1) {
  const config = {
    featConfig: {
      sampleRate: 16000,
      featureDim: 80,
    },
    modelConfig: {
      funasrNano: {
        encoderAdaptor: path.join(modelDir, 'encoder_adaptor.int8.onnx'),
        llm: path.join(modelDir, 'llm.int8.onnx'),
        embedding: path.join(modelDir, 'embedding.int8.onnx'),
        tokenizer: path.join(modelDir, 'Qwen3-0.6B'),
      },
      tokens: '',
      numThreads,
      provider: 'cpu',
      debug,
    },
  };

  return new sherpa_onnx.OfflineRecognizer(config);
}

/**
 * Read a waveform and create a stream for decoding.
 */
function createStreamFromFile(recognizer, file) {
  const wave = sherpa_onnx.readWave(file);
  const stream = recognizer.createStream();
  stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});
  return stream;
}

async function main() {
  const modelDir = './sherpa-onnx-funasr-nano-int8-2025-12-30';
  const recognizer = createRecognizer(modelDir);

  const testFiles = [
    'test_wavs/lyrics_en_1.wav',
    'test_wavs/lyrics_en_2.wav',
    'test_wavs/lyrics_en_3.wav',
  ].map(f => path.join(modelDir, f));

  // Create streams for each file
  const streams = testFiles.map(file => createStreamFromFile(recognizer, file));

  // Decode all streams concurrently
  const results =
      await Promise.all(streams.map(stream => recognizer.decodeAsync(stream)));

  console.log('Concurrent decode results:');
  testFiles.forEach((file, i) => {
    console.log(`${file}: ${results[i].text}`);
  });
}

main().catch(console.error);
