// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of
// one speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock takes the loudest
// 3 seconds in the first 15 seconds of the input as the speaker to keep. See
// ./test_offline_speech_enhancement_voxlock_with_voice_sample.js if you have a
// voice sample of the speaker.
//
// Please download the model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// Example:
//
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
//  tar xvf sherpa-onnx-voxlock-int8.tar.bz2

const sherpa_onnx = require('sherpa-onnx-node');

function createOfflineSpeechDenoiser() {
  const config = {
    model: {
      voxlock: {
        enrollModel: './sherpa-onnx-voxlock-int8/enroll.onnx',
        chunkModel: './sherpa-onnx-voxlock-int8/chunk_int8.onnx',
      },
      debug: true,
      numThreads: 1,
    },
  };

  return new sherpa_onnx.OfflineSpeechDenoiser(config);
}

const sd = createOfflineSpeechDenoiser();

const waveFilename = './sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav';
const wave = sherpa_onnx.readWave(waveFilename);

const begin = Date.now();
const denoised = sd.run({
  samples: wave.samples,
  sampleRate: wave.sampleRate,
  enableExternalBuffer: true
});
const elapsedSeconds = (Date.now() - begin) / 1000;

sherpa_onnx.writeWave(
    './voxlock-enhanced-16k.wav',
    {samples: denoised.samples, sampleRate: denoised.sampleRate});

const duration = wave.samples.length / wave.sampleRate;
const rtf = elapsedSeconds / duration;

console.log(`Saved to ./voxlock-enhanced-16k.wav`);
console.log(`Elapsed seconds: ${elapsedSeconds.toFixed(3)}`);
console.log(`Audio duration in seconds: ${duration.toFixed(3)}`);
console.log(
    `RTF: ${elapsedSeconds.toFixed(3)}/${duration.toFixed(3)} = ${rtf.toFixed(3)}`);
