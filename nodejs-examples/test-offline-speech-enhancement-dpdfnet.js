// Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Please download a DPDFNet model and ./inp_16k.wav used in this file
// from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
// or https://huggingface.co/Ceva-IP/DPDFNet
//
// This script shows how to use speech enhancement API from sherpa-onnx.
// Use dpdfnet_baseline.onnx, dpdfnet2.onnx, dpdfnet4.onnx, or dpdfnet8.onnx
// for 16 kHz downstream ASR or speech recognition.
// Use dpdfnet2_48khz_hr.onnx for 48 kHz enhancement output.
const sherpa_onnx = require('sherpa-onnx');

function createOfflineSpeechDenoiser() {
  const model = './dpdfnet2.onnx';
  let config = {
    model: {
      dpdfnet: {model},
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
