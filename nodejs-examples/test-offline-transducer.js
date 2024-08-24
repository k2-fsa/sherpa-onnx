// Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let modelConfig = {
    transducer: {
      encoder:
          './sherpa-onnx-zipformer-en-2023-06-26/encoder-epoch-99-avg-1.int8.onnx',
      decoder:
          './sherpa-onnx-zipformer-en-2023-06-26/decoder-epoch-99-avg-1.onnx',
      joiner:
          './sherpa-onnx-zipformer-en-2023-06-26/joiner-epoch-99-avg-1.int8.onnx',
    },
    tokens: './sherpa-onnx-zipformer-en-2023-06-26/tokens.txt',
    modelType: 'transducer',
  };

  let config = {
    modelConfig: modelConfig,
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}
const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename = './sherpa-onnx-zipformer-en-2023-06-26/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
