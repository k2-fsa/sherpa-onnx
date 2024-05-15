// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx-node');

// please download model files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
function createOfflineTts() {
  const config = {
    model: {
      vits: {
        model: './sherpa-onnx-vits-zh-ll/model.onnx',
        tokens: './sherpa-onnx-vits-zh-ll/tokens.txt',
        lexicon: './sherpa-onnx-vits-zh-ll/lexicon.txt',
        dictDir: './sherpa-onnx-vits-zh-ll/dict',
      },
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumStences: 1,
    ruleFsts:
        './sherpa-onnx-vits-zh-ll/date.fst,./sherpa-onnx-vits-zh-ll/phone.fst,./sherpa-onnx-vits-zh-ll/number.fst',
  };
  return new sherpa_onnx.OfflineTts(config);
}

const tts = createOfflineTts();

const text =
    '当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。2024年5月13号，拨打110或者18920240513。123456块钱。'

let start = Date.now();
const audio = tts.generate({text: text, sid: 2, speed: 1.0});
let stop = Date.now();
const elapsed_seconds = (stop - start) / 1000;
const duration = audio.samples.length / audio.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'secodns')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'secodns')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))

const filename = 'test-zh-ll.wav';
sherpa_onnx.writeWave(
    filename, {samples: audio.samples, sampleRate: audio.sampleRate});

console.log(`Saved to ${filename}`);
