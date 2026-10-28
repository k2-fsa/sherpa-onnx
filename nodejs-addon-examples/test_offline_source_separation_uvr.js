// Copyright (c)  2026  Nexmoe

const sherpa_onnx = require('sherpa-onnx-node');

// Download the model and test audio from:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/source-separation-models
const separator = new sherpa_onnx.OfflineSourceSeparation({
  model: {
    uvr: {model: './UVR-MDX-NET-Voc_FT.onnx'},
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
    const filename = `uvr-${name}-${channel}.wav`;
    sherpa_onnx.writeWave(filename, {samples, sampleRate: output.sampleRate});
    console.log(`Saved ${filename}`);
  });
}
