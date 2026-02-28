// Copyright (c)  2023-2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let modelConfig = {
    moonshine: {
      encoder:
          './sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort',
      mergedDecoder:
          './sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort',
    },
    tokens: './sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt',
  };

  let config = {
    modelConfig: modelConfig,
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename =
    './sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
