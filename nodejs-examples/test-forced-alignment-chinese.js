// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Forced alignment of Chinese speech with the sherpa-onnx Node.js API.
// It aligns the test wav of the zh-aishell3-pinyin model package with its
// transcript and writes a Praat TextGrid.
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
// tar xvf zh-aishell3-pinyin.tar.bz2
// rm zh-aishell3-pinyin.tar.bz2
const fs = require('fs');
const sherpa_onnx = require('sherpa-onnx');

function createForcedAligner() {
  const modelDir = './zh-aishell3-pinyin';
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

const waveFilename = './zh-aishell3-pinyin/zh.wav';
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform(wave.sampleRate, wave.samples);

// The transcript contains hanzi; the file must be saved as UTF-8.
const text = '不要问你的国家能为你做什么，而要问你能为你的国家做什么。';

const begin = Date.now();
const result = aligner.compute(stream, text);
const elapsedSeconds = (Date.now() - begin) / 1000;

for (const tier of result.tiers) {
  console.log(`tier ${tier.name}: ${tier.intervals.length} intervals`);
}

const outputFilename = 'zh.TextGrid';
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
