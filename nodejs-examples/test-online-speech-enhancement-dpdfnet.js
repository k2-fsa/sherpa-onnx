// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Please download a DPDFNet model and ./inp_16k.wav used in this file from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
// or https://huggingface.co/Ceva-IP/DPDFNet
//
// This script shows how to use the streaming speech enhancement API from
// sherpa-onnx.
const sherpa_onnx = require('sherpa-onnx');

function createOnlineSpeechDenoiser() {
  const model = './dpdfnet_baseline.onnx';
  const config = {
    model: {
      dpdfnet: {model},
      debug: 1,
    },
  };

  return sherpa_onnx.createOnlineSpeechDenoiser(config);
}

const speech_denoiser = createOnlineSpeechDenoiser();

const waveFilename = './inp_16k.wav';
const wave = sherpa_onnx.readWave(waveFilename);
const frameShift = speech_denoiser.frameShiftInSamples;
const output = [];

let start = 0;
while (start < wave.samples.length) {
  const end = Math.min(start + frameShift, wave.samples.length);
  const chunk = wave.samples.slice(start, end);
  const denoised = speech_denoiser.run(chunk, wave.sampleRate);
  output.push(...denoised.samples);
  start = end;
}

output.push(...speech_denoiser.flush().samples);

const outputFilename = './enhanced-online-dpdfnet.wav';
sherpa_onnx.writeWave(outputFilename, {
  samples: Float32Array.from(output),
  sampleRate: speech_denoiser.sampleRate,
});
console.log(`Saved to ${outputFilename}`);

speech_denoiser.free();
