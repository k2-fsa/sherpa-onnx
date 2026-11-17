// Copyright (c)  2026  Xiaomi Corporation
const fs = require('fs');
const sherpa_onnx = require('sherpa-onnx-node');

// Please download the model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
async function main() {
  const config = {
    'modelDir': './english-mfa-v3.1.0',
    'lexicon': './english-mfa-v3.1.0/lexicon.txt',
  };

  const aligner = await sherpa_onnx.ForcedAligner.createAsync(config);
  const stream = aligner.createStream();

  // Optional: per-stream options, e.g. debug output of one utterance.
  stream.setOption('debug', '1');

  const waveFilename = './english-mfa-v3.1.0/en.wav';
  const wave = sherpa_onnx.readWave(waveFilename);
  stream.acceptWaveform(wave);

  const text =
      'ask not what your country can do for you, ' +
      'ask what you can do for your country.';

  const start = Date.now();
  const result = await aligner.computeAsync(stream, text);
  const stop = Date.now();

  for (const tier of result.tiers) {
    console.log(`tier ${tier.name}: ${tier.intervals.length} intervals`);
  }

  const outputFilename = 'en-async.TextGrid';
  fs.writeFileSync(outputFilename, result.textgrid);

  const elapsed_seconds = (stop - start) / 1000;
  const duration = wave.samples.length / wave.sampleRate;
  const real_time_factor = elapsed_seconds / duration;
  console.log(`Saved to ${outputFilename}`);
  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));
}

main().catch(console.error);
