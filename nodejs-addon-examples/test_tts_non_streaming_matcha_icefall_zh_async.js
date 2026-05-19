// Copyright (c)  2026  Xiaomi Corporation
//
// Asynchronous text-to-speech with the Matcha Chinese model.
//
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
  const config = {
    model: {
      matcha: {
        acousticModel: './matcha-icefall-zh-baker/model-steps-3.onnx',
        vocoder: './vocos-22khz-univ.onnx',
        lexicon: './matcha-icefall-zh-baker/lexicon.txt',
        tokens: './matcha-icefall-zh-baker/tokens.txt',
      },
      debug: false,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumSentences: 1,
    ruleFsts:
        './matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst',
  };
  return await sherpa_onnx.OfflineTts.createAsync(config);
}

async function main() {
  const tts = await createOfflineTts();

  const text =
      '当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔. 某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。';

  const generationConfig = new sherpa_onnx.GenerationConfig({
    sid: 0,
    speed: 1.0,
    silenceScale: 0.2,
  });

  const start = Date.now();
  const audio = await tts.generateAsync({
    text,
    enableExternalBuffer: true,
    generationConfig,
    onProgress: ({samples, progress}) => {
      process.stdout.write(
          `Progress: ${(progress * 100).toFixed(1)}%, ` +
          `Samples: ${samples.length}\r`);
      return 1;
    },
  });

  console.log('');
  const stop = Date.now();
  const elapsed_seconds = (stop - start) / 1000;
  const duration = audio.samples.length / audio.sampleRate;
  const real_time_factor = elapsed_seconds / duration;
  console.log('Wave duration', duration.toFixed(3), 'seconds');
  console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
  console.log(
      `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
      real_time_factor.toFixed(3));

  const filename = 'test-matcha-zh-async.wav';
  sherpa_onnx.writeWave(
      filename, {samples: audio.samples, sampleRate: audio.sampleRate});
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exitCode = 1;
});
