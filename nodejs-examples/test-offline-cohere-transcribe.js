// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  const config = {
    modelConfig: {
      cohereTranscribe: {
        encoder:
            './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx',
        decoder:
            './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx',
        usePunct: 1,
        useItn: 1,
      },
      tokens:
          './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt',
    }
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();
stream.setOption('language', 'en');

const waveFilename =
    './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
