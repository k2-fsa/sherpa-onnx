// Copyright (c)  2026  Nexmoe

const sherpa_onnx = require('sherpa-onnx-node');

// Download the models and test audio from:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models
const separator = new sherpa_onnx.OfflineSourceSeparation({
  model: {
    spleeter: {
      vocals: './sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx',
      accompaniment:
          './sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx',
    },
    numThreads: 1,
  },
});

// The model expects stereo; readWave() would keep only the first channel.
const wave = sherpa_onnx.readWaveMultiChannel('./qi-feng-le-zh.wav');

const start = Date.now();
const output = separator.process({
  samples: wave.samples,
  sampleRate: wave.sampleRate,
  enableExternalBuffer: false,
});
const end = Date.now();

// Each stem contains one channel per input channel; writeWaveMultiChannel
// writes a stem to a single wave file.
for (let i = 0; i < output.stems.length; ++i) {
  const filename = ['spleeter-vocals.wav', 'spleeter-non-vocals.wav'][i];
  sherpa_onnx.writeWaveMultiChannel(
      filename, {samples: output.stems[i], sampleRate: output.sampleRate});
  console.log(`Saved ${filename}`);
}

const elapsed_seconds = (end - start) / 1000;
const audio_duration = wave.samples[0].length / wave.sampleRate;
const real_time_factor = elapsed_seconds / audio_duration;

console.log(`Elapsed seconds: ${elapsed_seconds.toFixed(3)}`);
console.log(`Audio duration in seconds: ${audio_duration.toFixed(3)}`);
console.log(
    `RTF: ${elapsed_seconds.toFixed(3)}/${audio_duration.toFixed(3)} =`,
    real_time_factor.toFixed(3));
