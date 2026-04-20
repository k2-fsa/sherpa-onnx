// Copyright (c)  2026  Xiaomi Corporation
//  This file shows how to use the async API to decode multiple files
const path = require('path');
const sherpa_onnx = require('sherpa-onnx-node');

/**
 * Create an OfflineRecognizer with Cohere Transcribe asynchronously.
 */
async function createRecognizerAsync(modelDir, numThreads = 2, debug = 0) {
  const config = {
    featConfig: {
      sampleRate: 16000,
      featureDim: 80,
    },
    modelConfig: {
      cohereTranscribe: {
        encoder: path.join(modelDir, 'encoder.int8.onnx'),
        decoder: path.join(modelDir, 'decoder.int8.onnx'),
        usePunct: 1,
        useItn: 1,
      },
      tokens: path.join(modelDir, 'tokens.txt'),
      numThreads,
      provider: 'cpu',
      debug,
    },
  };

  return await sherpa_onnx.OfflineRecognizer.createAsync(config);
}

/**
 * Read a waveform and create a stream for decoding.
 */
function createStreamFromFile(recognizer, file, language) {
  const wave = sherpa_onnx.readWave(file);
  const stream = recognizer.createStream();
  stream.setOption('language', language);
  stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});
  return stream;
}

async function main() {
  const modelDir = './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01';

  const recognizer = await createRecognizerAsync(modelDir);

  const testFiles = [
    {
      language: 'en',
      file: path.join(modelDir, 'test_wavs/en.wav'),
    },
    {
      language: 'fr',
      file: path.join(modelDir, 'test_wavs/fr.wav'),
    },
    {
      language: 'zh',
      file: path.join(modelDir, 'test_wavs/zh.wav'),
    },
    {
      language: 'de',
      file: path.join(modelDir, 'test_wavs/de.wav'),
    },
    {
      language: 'es',
      file: path.join(modelDir, 'test_wavs/es.wav'),
    },
    {
      language: 'ja',
      file: path.join(modelDir, 'test_wavs/ja.wav'),
    },
    {
      language: 'ko',
      file: path.join(modelDir, 'test_wavs/ko.wav'),
    },
    {
      language: 'vi',
      file: path.join(modelDir, 'test_wavs/vi.wav'),
    },
    {
      language: 'ar',
      file: path.join(modelDir, 'test_wavs/ar.wav'),
    },
  ];

  const streams = testFiles.map(
      entry => createStreamFromFile(recognizer, entry.file, entry.language));

  const results =
      await Promise.all(streams.map(stream => recognizer.decodeAsync(stream)));

  console.log('Concurrent decode results:');
  testFiles.forEach((entry, i) => {
    console.log(`${entry.language} ${entry.file}: ${results[i].text}`);
  });
}

main().catch(console.error);
