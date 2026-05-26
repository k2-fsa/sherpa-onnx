// Copyright (c)  2026  Xiaomi Corporation
//  This file shows how to use the async API to decode multiple files
const path = require('path');
const sherpa_onnx = require('sherpa-onnx-node');

/**
 * Create an OfflineRecognizer with Qwen3 ASR model asynchronously.
 */
async function createRecognizerAsync(modelDir, numThreads = 2, debug = 1) {
  const config = {
    featConfig: {
      sampleRate: 16000,
      featureDim: 80,
    },
    modelConfig: {
      qwen3Asr: {
        convFrontend: path.join(modelDir, 'conv_frontend.onnx'),
        encoder: path.join(modelDir, 'encoder.int8.onnx'),
        decoder: path.join(modelDir, 'decoder.int8.onnx'),
        tokenizer: path.join(modelDir, 'tokenizer'),
        hotwords: '',
      },
      tokens: '',
      numThreads,
      provider: 'cpu',
      debug,
    },
  };

  // Use the async C++ API to create recognizer without blocking Node.js
  return await sherpa_onnx.OfflineRecognizer.createAsync(config);
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
  const modelDir = './sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25';

  // Async recognizer creation
  const recognizer = await createRecognizerAsync(modelDir);

  const testFiles = [
    'test_wavs/raokouling.wav',
    'test_wavs/fast1.wav',
    'test_wavs/f1_noise.wav',
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
