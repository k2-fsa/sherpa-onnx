// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock takes the loudest
// 3 seconds in the first 15 seconds of the input as the speaker to keep. See
// ./test-offline-speech-enhancement-voxlock-with-voice-sample.js if you have a
// voice sample of the speaker.
//
// Please download the model used in this file from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// Example:
//
//  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
//  tar xvf sherpa-onnx-voxlock-int8.tar.bz2
//
// This script shows how to use speech enhancement API from sherpa-onnx.
const sherpa_onnx = require('sherpa-onnx');

function createOfflineSpeechDenoiser() {
  const config = {
    model: {
      voxlock: {
        enrollModel: './sherpa-onnx-voxlock-int8/enroll.onnx',
        chunkModel: './sherpa-onnx-voxlock-int8/chunk_int8.onnx',
      },
      debug: 1,
    },
  };

  return sherpa_onnx.createOfflineSpeechDenoiser(config);
}

const speech_denoiser = createOfflineSpeechDenoiser();

const waveFilename = './sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav';
const wave = sherpa_onnx.readWave(waveFilename);

const begin = Date.now();
const denoised = speech_denoiser.run(wave.samples, wave.sampleRate);
const elapsedSeconds = (Date.now() - begin) / 1000;

const outputFilename = './voxlock-enhanced-16k.wav';
sherpa_onnx.writeWave(outputFilename, denoised);

const duration = wave.samples.length / wave.sampleRate;
const rtf = elapsedSeconds / duration;

console.log(`Saved to ${outputFilename}`);
console.log(`Elapsed seconds: ${elapsedSeconds.toFixed(3)}`);
console.log(`Audio duration in seconds: ${duration.toFixed(3)}`);
console.log(
    `RTF: ${elapsedSeconds.toFixed(3)}/${duration.toFixed(3)} = ${rtf.toFixed(3)}`);

speech_denoiser.free();
