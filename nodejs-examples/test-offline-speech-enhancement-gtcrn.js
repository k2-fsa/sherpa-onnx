// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Please download ./gtcrn_simple.onnx and ./inp_16k.wav used in this file
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// This script shows how to use speech enhancement API from sherpa-onnx.
const sherpa_onnx = require('sherpa-onnx');

function createOfflineSpeechDenoiser() {
  let config = {
    model: {
      gtcrn: {model: './gtcrn_simple.onnx'},
      debug: 1,
    },
  };

  return sherpa_onnx.createOfflineSpeechDenoiser(config);
}

speech_denoiser = createOfflineSpeechDenoiser();

const waveFilename = './inp_16k.wav';
const wave = sherpa_onnx.readWave(waveFilename);

const denoised = speech_denoiser.run(wave.samples, wave.sampleRate);
sherpa_onnx.writeWave('./enhanced-16k.wav', denoised);
console.log('Saved to ./enhanced-16k.wav');

speech_denoiser.free();
