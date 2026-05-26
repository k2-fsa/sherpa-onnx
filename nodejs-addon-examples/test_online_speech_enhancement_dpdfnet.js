// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx-node');

function createOnlineSpeechDenoiser() {
  const config = {
    model: {
      dpdfnet: {model: './dpdfnet_baseline.onnx'},
      debug: true,
      numThreads: 1,
    },
  };

  return new sherpa_onnx.OnlineSpeechDenoiser(config);
}

const sd = createOnlineSpeechDenoiser();
const wave = sherpa_onnx.readWave('./inp_16k.wav');
const output = [];
const frameShift = sd.frameShiftInSamples;

for (let start = 0; start < wave.samples.length; start += frameShift) {
  const end = Math.min(start + frameShift, wave.samples.length);
  const chunk = wave.samples.slice(start, end);
  const denoised = sd.run({
    samples: chunk,
    sampleRate: wave.sampleRate,
    enableExternalBuffer: true
  });
  output.push(...denoised.samples);
}

const tail = sd.flush(true);
output.push(...tail.samples);

sherpa_onnx.writeWave(
    './enhanced-online-dpdfnet.wav',
    {samples: Float32Array.from(output), sampleRate: sd.sampleRate});

console.log('Saved to ./enhanced-online-dpdfnet.wav');
