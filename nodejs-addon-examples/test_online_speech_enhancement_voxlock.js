// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of
// one speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock passes the audio
// through unchanged until it has heard 3 seconds of speech, takes that speech
// as the speaker to keep, and isolates from then on. See
// ./test_online_speech_enhancement_voxlock_with_voice_sample.js if you have a
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

function createOnlineSpeechDenoiser() {
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

  return new sherpa_onnx.OnlineSpeechDenoiser(config);
}

const sd = createOnlineSpeechDenoiser();
const wave =
    sherpa_onnx.readWave('./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav');
const output = [];
const frameShift = sd.frameShiftInSamples;

const begin = Date.now();

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

const elapsedSeconds = (Date.now() - begin) / 1000;

sherpa_onnx.writeWave(
    './voxlock-enhanced-online-16k.wav',
    {samples: Float32Array.from(output), sampleRate: sd.sampleRate});

const duration = wave.samples.length / wave.sampleRate;
const rtf = elapsedSeconds / duration;

console.log(`Saved to ./voxlock-enhanced-online-16k.wav`);
console.log(`Elapsed seconds: ${elapsedSeconds.toFixed(3)}`);
console.log(`Audio duration in seconds: ${duration.toFixed(3)}`);
console.log(
    `RTF: ${elapsedSeconds.toFixed(3)}/${duration.toFixed(3)} = ${rtf.toFixed(3)}`);
