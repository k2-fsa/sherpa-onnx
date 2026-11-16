// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Forced alignment of English speech with the sherpa-onnx Node.js API.
// It aligns the test wav of the english-mfa-v3.1.0 model package with its
// transcript and writes a Praat TextGrid.
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
// tar xvf english-mfa-v3.1.0.tar.bz2
// rm english-mfa-v3.1.0.tar.bz2
const fs = require('fs');
const sherpa_onnx = require('sherpa-onnx');

function createForcedAligner() {
  const modelDir = './english-mfa-v3.1.0';
  const config = {
    modelDir: modelDir,
    lexicon: modelDir + '/lexicon.txt',
  };

  return sherpa_onnx.createForcedAligner(config);
}

const aligner = createForcedAligner();
const stream = aligner.createStream();

// Optional: per-stream options, e.g. debug output of one utterance.
stream.setOption('debug', '1');

const waveFilename = './english-mfa-v3.1.0/en.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

const text =
    'ask not what your country can do for you, ' +
    'ask what you can do for your country.';

const begin = Date.now();
const result = aligner.compute(stream, text);
const elapsedSeconds = (Date.now() - begin) / 1000;

for (const tier of result.tiers) {
  console.log(`tier ${tier.name}: ${tier.intervals.length} intervals`);
}

const outputFilename = 'en.TextGrid';
fs.writeFileSync(outputFilename, result.textgrid);

const duration = wave.samples.length / wave.sampleRate;
const rtf = elapsedSeconds / duration;

console.log(`Saved to ${outputFilename}`);
console.log(`Elapsed seconds: ${elapsedSeconds.toFixed(3)}`);
console.log(`Audio duration in seconds: ${duration.toFixed(3)}`);
console.log(
    `RTF: ${elapsedSeconds.toFixed(3)}/${duration.toFixed(3)} = ${rtf.toFixed(3)}`);

stream.free();
aligner.free();
