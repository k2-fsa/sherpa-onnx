// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Please download a speech enhancement model and ./inp_16k.wav used in this file
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// This script shows how to use speech enhancement API from sherpa-onnx.
const sherpa_onnx = require('sherpa-onnx');

function createOfflineSpeechDenoiser() {
  const model = './gtcrn_simple.onnx';
  let config = {
    model: {
      gtcrn: {model},
      debug: 1,
    },
  };

  return sherpa_onnx.createOfflineSpeechDenoiser(config);
}

const speech_denoiser = createOfflineSpeechDenoiser();

const waveFilename = './inp_16k.wav';
const wave = sherpa_onnx.readWave(waveFilename);

const denoised = speech_denoiser.run(wave.samples, wave.sampleRate);
const outputFilename = './enhanced.wav';
sherpa_onnx.writeWave(outputFilename, denoised);
console.log(`Saved to ${outputFilename}`);

speech_denoiser.free();
