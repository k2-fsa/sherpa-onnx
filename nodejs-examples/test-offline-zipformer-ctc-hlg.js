// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// This file shows how to use a non-streaming Zipformer CTC model with an HLG
// decoding graph via sherpa-onnx's Node API.
//
// The HLG graph is passed to the C API through ctcFstDecoderConfig.
const fs = require('fs');
const {Readable} = require('stream');
const wav = require('wav');

const sherpa_onnx = require('sherpa-onnx');

function createOfflineRecognizer() {
  let config = {
    modelConfig: {
      zipformerCtc: {
        model: './sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx',
      },
      tokens: './sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt',
    },
    // This is what the example exercises: the graph reaches the C API through
    // ctcFstDecoderConfig.
    ctcFstDecoderConfig: {
      graph: './sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst',
      maxActive: 3000,
    },
  };

  return sherpa_onnx.createOfflineRecognizer(config);
}

const recognizer = createOfflineRecognizer();
const stream = recognizer.createStream();

const waveFilename = './sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

recognizer.decode(stream);
const text = recognizer.getResult(stream).text;
console.log(text);

stream.free();
recognizer.free();
