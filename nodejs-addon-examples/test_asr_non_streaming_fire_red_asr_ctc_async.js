// Copyright (c)  2026  Xiaomi Corporation
//  This file shows how to use the async API to decode multiple files
const path = require('path');
const sherpa_onnx = require('sherpa-onnx-node');

/**
 * Create an OfflineRecognizer with FireRedASR CTC model asynchronously.
 */
async function createRecognizerAsync(numThreads = 2, debug = 1) {
  const config = {
    featConfig: {
      sampleRate: 16000,
      featureDim: 80,
    },
    modelConfig: {
      fireRedAsrCtc: {
        model:
            './sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx',
      },
      tokens:
          './sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt',
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
  const modelDir = './sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25';

  // Async recognizer creation
  const recognizer = await createRecognizerAsync(modelDir);

  const testFiles = [
    'test_wavs/0.wav',
    'test_wavs/1.wav',
    'test_wavs/2.wav',
    'test_wavs/3-sichuan.wav',
    'test_wavs/3.wav',
    'test_wavs/4-tianjin.wav',
    'test_wavs/5-henan.wav',
    'test_wavs/8k.wav',
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
