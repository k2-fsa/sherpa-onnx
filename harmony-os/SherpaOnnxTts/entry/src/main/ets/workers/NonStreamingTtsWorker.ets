import worker, { ThreadWorkerGlobalScope, MessageEvents, ErrorEvent } from '@ohos.worker';

import { fileIo as fs } from '@kit.CoreFileKit';

import {OfflineTtsConfig, OfflineTts, listRawfileDir, TtsInput, TtsOutput} from 'sherpa_onnx';
import { buffer } from '@kit.ArkTS';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

let tts: OfflineTts;
let cancelled = false;

function mkdir(context: Context, parts: string[]) {
  const path = parts.join('/');
  if (fs.accessSync(path)) {
    return;
  }

  const sandboxPath: string = context.getApplicationContext().filesDir;
  let d = sandboxPath
  for (const p of parts) {
    d = d + '/' + p;

    if (fs.accessSync(d)) {
      continue;
    }

    fs.mkdirSync(d);
  }
}

function copyRawFileDirToSandbox(context: Context, srcDir: string) {
  let mgr = context.resourceManager;
  const allFiles: string[] = listRawfileDir(mgr, srcDir);
  for (const src of allFiles) {
    const parts: string[] = src.split('/');
    if (parts.length != 1) {
      mkdir(context, parts.slice(0, -1));
    }

    copyRawFileToSandbox(context, src, src);
  }
}

function copyRawFileToSandbox(context: Context, src: string, dst: string) {
  // see https://blog.csdn.net/weixin_44640245/article/details/142634846
  // https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/rawfile-guidelines-V5
  let uint8Array: Uint8Array = context.resourceManager.getRawFileContentSync(src);

  // https://developer.huawei.com/consumer/cn/doc/harmonyos-references-V5/js-apis-file-fs-V5#fsmkdir
  let sandboxPath: string = context.getApplicationContext().filesDir;
  let filepath = sandboxPath + '/'  + dst;

  if (fs.accessSync(filepath)) {
    // if the destination exists and has the expected file size,
    // then we skip copying it
    let stat = fs.statSync(filepath);
    if (stat.size == uint8Array.length) {
      return;
    }
  }

  const fp = fs.openSync(filepath, fs.OpenMode.WRITE_ONLY | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
  fs.writeSync(fp.fd, buffer.from(uint8Array).buffer)
  fs.close(fp.fd);
}

function initTts(context: Context): OfflineTts {
  // Such a design is to make it easier to build flutter APPs with
  // github actions for a variety of tts models
  //
  // See https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/flutter/generate-tts.py
  // for details

  let modelDir = '';

  // for VITS begin
  let modelName = '';
  // for VITS end

  // for Matcha begin
  let acousticModelName = '';
  let vocoder = '';
  // for Matcha end

  let ruleFsts = '';
  let ruleFars = '';
  let lexicon = '';
  let dataDir = '';
  let dictDir = '';
  // You can select an example below and change it according to match your
  // selected tts model

  // ============================================================
  // Your change starts here
  // ============================================================

  // Example 1:
  // modelDir = 'vits-vctk';
  // modelName = 'vits-vctk.onnx';
  // lexicon = 'lexicon.txt';

  // Example 2:
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  // modelDir = 'vits-piper-en_US-amy-low';
  // modelName = 'en_US-amy-low.onnx';
  // dataDir = 'espeak-ng-data';

  // Example 3:
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
  // modelDir = 'vits-icefall-zh-aishell3';
  // modelName = 'model.onnx';
  // ruleFsts = 'phone.fst,date.fst,number.fst,new_heteronym.fst';
  // ruleFars = 'rule.far';
  // lexicon = 'lexicon.txt';

  // Example 4:
  // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#csukuangfj-vits-zh-hf-fanchen-c-chinese-187-speakers
  // modelDir = 'vits-zh-hf-fanchen-C';
  // modelName = 'vits-zh-hf-fanchen-C.onnx';
  // lexicon = 'lexicon.txt';
  // dictDir = 'dict';

  // Example 5:
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
  // modelDir = 'vits-coqui-de-css10';
  // modelName = 'model.onnx';

  // Example 6
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
  // modelDir = 'vits-piper-en_US-libritts_r-medium';
  // modelName = 'en_US-libritts_r-medium.onnx';
  // dataDir = 'espeak-ng-data';

  // Example 7
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
  // modelDir = 'vits-melo-tts-zh_en';
  // modelName = 'model.onnx';
  // lexicon = 'lexicon.txt';
  // dictDir = 'dict';
  // ruleFsts = `date.fst,phone.fst,number.fst`;

  // Example 8
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-zh-baker-chinese-1-female-speaker
  // modelDir = 'matcha-icefall-zh-baker'
  // acousticModelName = 'model-steps-3.onnx'
  // vocoder = 'hifigan_v2.onnx'
  // lexicon = 'lexicon.txt'
  // dictDir = 'dict';
  // ruleFsts = `date.fst,phone.fst,number.fst`;

  // Example 9
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker
  // modelDir = 'matcha-icefall-en_US-ljspeech'
  // acousticModelName = 'model-steps-3.onnx'
  // vocoder = 'hifigan_v2.onnx'
  // dataDir = 'espeak-ng-data';

  // ============================================================
  // Please don't change the remaining part of this function
  // ============================================================

  if (modelName == '' && acousticModelName == '' && vocoder == '') {
    throw new Error('You are supposed to select a model by changing the code before you run the app');
  }

  if (modelName != '' && acousticModelName != '') {
    throw new Error('Please select either VITS or Matcha, not both');
  }

  if (acousticModelName != '' && vocoder == '') {
    throw new Error('Please provider vocoder for matcha tts models');
  }

  if (modelName != '') {
    modelName = modelDir + '/' + modelName;
  }

  if (acousticModelName != '') {
    acousticModelName = modelDir + '/' + acousticModelName;
  }

  if (ruleFsts != '') {
    let fsts = ruleFsts.split(',')
    let tmp: string[] = [];
    for (const f of fsts) {
      tmp.push(modelDir + '/' + f);
    }
    ruleFsts = tmp.join(',');
  }

  if (ruleFars != '') {
    let fars = ruleFars.split(',')
    let tmp: string[] = [];
    for (const f of fars) {
      tmp.push(modelDir + '/' + f);
    }
    ruleFars = tmp.join(',');
  }

  if (lexicon != '') {
    lexicon = modelDir + '/' + lexicon;
  }

  if (dataDir != '') {
    copyRawFileDirToSandbox(context, modelDir + '/' + dataDir)
    let sandboxPath: string = context.getApplicationContext().filesDir;
    dataDir = sandboxPath + '/' + modelDir + '/'  + dataDir;
  }

  if (dictDir != '') {
    copyRawFileDirToSandbox(context, modelDir + '/' + dictDir)
    let sandboxPath: string = context.getApplicationContext().filesDir;
    dictDir = sandboxPath + '/' + modelDir + '/'  + dictDir;
  }

  const tokens = modelDir + '/tokens.txt';

  const config: OfflineTtsConfig = new OfflineTtsConfig();
  config.model.vits.model = modelName;
  config.model.vits.lexicon = lexicon;
  config.model.vits.tokens = tokens;
  config.model.vits.dataDir = dataDir;
  config.model.vits.dictDir = dictDir;

  config.model.matcha.acousticModel = acousticModelName;
  config.model.matcha.vocoder = vocoder;
  config.model.matcha.lexicon = lexicon;
  config.model.matcha.tokens = tokens;
  config.model.matcha.dataDir = dataDir;
  config.model.matcha.dictDir = dictDir;

  config.model.numThreads = 2;
  config.model.debug = true;
  config.ruleFsts = ruleFsts;
  config.ruleFars = ruleFars;

  return new OfflineTts(config, context.resourceManager);
}

interface TtsCallbackData {
  samples: Float32Array;
  progress: number;
}

function callback(data: TtsCallbackData): number {
  workerPort.postMessage({
    'msgType': 'tts-generate-partial',
    samples: Float32Array.from(data.samples),
    progress: data.progress,
  });

  // 0 means to stop generating in C++
  // 1 means to continue generating in C++
  return cancelled? 0 : 1;
}

/**
 * Defines the event handler to be called when the worker thread receives a message sent by the host thread.
 * The event handler is executed in the worker thread.
 *
 * @param e message data
 */
workerPort.onmessage = (e: MessageEvents) => {
  const msgType = e.data['msgType'] as string;
  console.log(`msg-type: ${msgType}`);
  if (msgType == 'init-tts' && !tts) {
    const context = e.data['context'] as Context;
    tts = initTts(context);
    workerPort.postMessage({ 'msgType': 'init-tts-done',
      sampleRate: tts.sampleRate,
      numSpeakers: tts.numSpeakers,
    });
  }

  if (msgType == 'tts-generate-cancel') {
    cancelled = true;
  }

  if (msgType == 'tts-generate') {
    const text = e.data['text'] as string;
    console.log(`recevied text ${text}`);
    const input: TtsInput = new TtsInput();
    input.text = text;
    input.sid = e.data['sid'] as number;
    input.speed = e.data['speed'] as number;
    input.callback = callback;

    cancelled = false;
    if (true) {
      tts.generateAsync(input).then((ttsOutput: TtsOutput) => {
        console.log(`sampleRate: ${ttsOutput.sampleRate}`);

        workerPort.postMessage({
          'msgType': 'tts-generate-done',
          samples: Float32Array.from(ttsOutput.samples),
        });

      });
    } else {
      const ttsOutput: TtsOutput = tts.generate(input);
      workerPort.postMessage({
        'msgType': 'tts-generate-done',
        samples: Float32Array.from(ttsOutput.samples),
      });
    }


  }
}

/**
 * Defines the event handler to be called when the worker receives a message that cannot be deserialized.
 * The event handler is executed in the worker thread.
 *
 * @param e message data
 */
workerPort.onmessageerror = (e: MessageEvents) => {
}

/**
 * Defines the event handler to be called when an exception occurs during worker execution.
 * The event handler is executed in the worker thread.
 *
 * @param e error message
 */
workerPort.onerror = (e: ErrorEvent) => {
}
