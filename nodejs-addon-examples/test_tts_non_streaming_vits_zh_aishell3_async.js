// Copyright (c)  2026  Xiaomi Corporation
//
// Asynchronous text-to-speech with the VITS Chinese (AiShell3) model.
//
const sherpa_onnx = require('sherpa-onnx-node');

async function createOfflineTts() {
  const config = {
    model: {
      vits: {
        model: './vits-icefall-zh-aishell3/model.onnx',
        tokens: './vits-icefall-zh-aishell3/tokens.txt',
        lexicon: './vits-icefall-zh-aishell3/lexicon.txt',
      },
      debug: false,
      numThreads: 1,
      provider: 'cpu',
    },
    maxNumSentences: 1,
    ruleFsts:
        './vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/number.fst,./vits-icefall-zh-aishell3/new_heteronym.fst',
    ruleFars: './vits-icefall-zh-aishell3/rule.far',
  };
  return await sherpa_onnx.OfflineTts.createAsync(config);
}

async function main() {
  const tts = await createOfflineTts();

  const text =
      '他在长沙出生，长白山长大，去过长江，现在他是一个银行的行长，主管行政工作。有困难，请拨110，或者13020240513。今天是2024年5月13号, 他上个月的工资是12345块钱。';

  const generationConfig = new sherpa_onnx.GenerationConfig({
    sid: 88,
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

  const filename = 'test-vits-zh-aishell3-async.wav';
  sherpa_onnx.writeWave(
      filename, {samples: audio.samples, sampleRate: audio.sampleRate});
  console.log(`Saved to ${filename}`);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exitCode = 1;
});
