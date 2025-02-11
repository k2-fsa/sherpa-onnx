// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// please download model files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
function createOfflineTts() {
  const config = {
    model: {
      vits: {
        model: './vits-icefall-zh-aishell3/model.onnx',
        tokens: './vits-icefall-zh-aishell3/tokens.txt',
        lexicon: './vits-icefall-zh-aishell3/lexicon.txt',
      },
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumSentences: 1,
    ruleFsts:
        './vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/number.fst,./vits-icefall-zh-aishell3/new_heteronym.fst',
    ruleFars: './vits-icefall-zh-aishell3/rule.far',
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();

const text =
    '他在长沙出生，长白山长大，去过长江，现在他是一个银行的行长，主管行政工作。有困难，请拨110，或者13020240513。今天是2024年5月13号, 他上个月的工资是12345块钱。'

let start = Date.now();
const audio = tts.generate({text: text, sid: 88, speed: 1.0});
let stop = Date.now();
const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'secodns')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'secodns')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))

const filename = 'test-zh-aishell3.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
