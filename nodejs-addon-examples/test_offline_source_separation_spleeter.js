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

const wave = sherpa_onnx.readWave('./qi-feng-le-zh.wav');
const output = separator.process({
  samples: [wave.samples],
  sampleRate: wave.sampleRate,
  enableExternalBuffer: false,
});

// writeWave writes mono audio. Each stem contains separate stereo channels.
for (let i = 0; i < output.stems.length; ++i) {
  const name = ['vocals', 'accompaniment'][i];
  output.stems[i].forEach((samples, channel) => {
    const filename = `spleeter-${name}-${channel}.wav`;
    sherpa_onnx.writeWave(filename, {samples, sampleRate: output.sampleRate});
    console.log(`Saved ${filename}`);
  });
}
