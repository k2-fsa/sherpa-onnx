// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock passes the audio
// through unchanged until it has heard 3 seconds of speech, takes that speech
// as the speaker to keep, and isolates from then on. See
// ./test-online-speech-enhancement-voxlock-with-voice-sample.js if you have a
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
// This script shows how to use the streaming speech enhancement API from
// sherpa-onnx.
const sherpa_onnx = require('sherpa-onnx');

function createOnlineSpeechDenoiser() {
  const config = {
    model: {
      voxlock: {
        enrollModel: './sherpa-onnx-voxlock-int8/enroll.onnx',
        chunkModel: './sherpa-onnx-voxlock-int8/chunk_int8.onnx',
      },
      debug: 1,
    },
  };

  return sherpa_onnx.createOnlineSpeechDenoiser(config);
}

const speech_denoiser = createOnlineSpeechDenoiser();

const waveFilename = './sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav';
const wave = sherpa_onnx.readWave(waveFilename);
const frameShift = speech_denoiser.frameShiftInSamples;
const output = [];

const begin = Date.now();

let start = 0;
while (start < wave.samples.length) {
  const end = Math.min(start + frameShift, wave.samples.length);
  const chunk = wave.samples.slice(start, end);
  const denoised = speech_denoiser.run(chunk, wave.sampleRate);
  output.push(...denoised.samples);
  start = end;
}

output.push(...speech_denoiser.flush().samples);

const elapsedSeconds = (Date.now() - begin) / 1000;

const outputFilename = './voxlock-enhanced-online-16k.wav';
sherpa_onnx.writeWave(outputFilename, {
  samples: Float32Array.from(output),
  sampleRate: speech_denoiser.sampleRate,
});

const duration = wave.samples.length / wave.sampleRate;
const rtf = elapsedSeconds / duration;

console.log(`Saved to ${outputFilename}`);
console.log(`Elapsed seconds: ${elapsedSeconds.toFixed(3)}`);
console.log(`Audio duration in seconds: ${duration.toFixed(3)}`);
console.log(
    `RTF: ${elapsedSeconds.toFixed(3)}/${duration.toFixed(3)} = ${rtf.toFixed(3)}`);

speech_denoiser.free();
