import { ErrorEvent, MessageEvents, ThreadWorkerGlobalScope, worker } from '@kit.ArkTS';
import {
  OfflineRecognizer,
  OfflineRecognizerConfig,
  OfflineStream,
  OnlineRecognizerResult,
  readWaveFromBinary,
  SileroVadConfig,
  SpeechSegment,
  Vad,
  VadConfig,
} from 'sherpa_onnx';
import { Context } from '@kit.AbilityKit';
import { fileIo } from '@kit.CoreFileKit';
import { getOfflineModelConfig } from '../pages/NonStreamingAsrModels';
import { BusinessError } from '@kit.BasicServicesKit';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

let recognizer: OfflineRecognizer;
let vad: Vad; // vad for decoding files
let vadMic: Vad; // vad for mic

function initVad(context: Context): Vad {
  let mgr = context.resourceManager;
  const config: VadConfig = new VadConfig(
    new SileroVadConfig(
      'silero_vad.onnx',
      0.5,
      0.25,
      0.5,
      512,
    ),
    16000,
    true,
    1,
  );

  const bufferSizeInSeconds = 60;
  return new Vad(config, bufferSizeInSeconds, mgr);
}

function initNonStreamingAsr(context: Context): OfflineRecognizer {
  let mgr = context.resourceManager;
  const config: OfflineRecognizerConfig = new OfflineRecognizerConfig();

  // Note that you can switch to a new model by changing type
  //
  // If you use type = 2, which means you will use
  // sherpa-onnx-whisper-tiny.en
  // we assume you have the following folder structure in you resources/rawfile
  /*
  (py38) fangjuns-MacBook-Pro:main fangjun$ pwd
  /Users/fangjun/open-source/sherpa-onnx/harmony-os/SherpaOnnxVadAsr/entry/src/main
  (py38) fangjuns-MacBook-Pro:main fangjun$ tree resources/rawfile/
  resources/rawfile/
  ├── sherpa-onnx-whisper-tiny.en
  │   ├── README.md
  │   ├── tiny.en-decoder.int8.onnx
  │   ├── tiny.en-encoder.int8.onnx
  │   └── tiny.en-tokens.txt
  └── silero_vad.onnx

  1 directory, 5 files
   */
  const type = 2;
  config.modelConfig = getOfflineModelConfig(type);
  config.modelConfig.debug = true;
  config.ruleFsts = '';
  return new OfflineRecognizer(config, mgr);
}

interface Wave {
  samples: Float32Array;
  sampleRate: number;
}

function decodeFile(filename: string): string {
  vad.reset();

  const fp = fileIo.openSync(filename);
  const stat = fileIo.statSync(fp.fd);
  const arrayBuffer = new ArrayBuffer(stat.size);
  fileIo.readSync(fp.fd, arrayBuffer);
  const data: Uint8Array = new Uint8Array(arrayBuffer);

  const wave: Wave = readWaveFromBinary(data);
  if (wave.sampleRate != 16000) {
    return `the sample rate in ${filename} is not 16000Hz. Given: ${wave.sampleRate}Hz.\nPlease select a wav file of 16kHz.`;
  }

  console.log(`sample rate ${wave.sampleRate}`);
  console.log(`samples length ${wave.samples.length}`);
  const resultList: string[] = [];

  const windowSize: number = vad.config.sileroVad.windowSize;
  for (let i = 0; i < wave.samples.length; i += windowSize) {
    const thisWindow: Float32Array = wave.samples.subarray(i, i + windowSize)
    vad.acceptWaveform(thisWindow);
    if (i + windowSize >= wave.samples.length) {
      vad.flush();
    }
    while (!vad.isEmpty()) {
      const segment: SpeechSegment = vad.front();
      const _startTime: number = (segment.start / wave.sampleRate);
      const _endTime: number = _startTime + segment.samples.length / wave.sampleRate;

      if (_endTime - _startTime < 0.2) {
        vad.pop();
        continue;
      }

      const startTime: string = _startTime.toFixed(2);
      const endTime: string = _endTime.toFixed(2);

      const progress: number = (segment.start + segment.samples.length) / wave.samples.length * 100;

      workerPort.postMessage({ 'msgType': 'non-streaming-asr-vad-decode-progress', progress });

      const stream: OfflineStream = recognizer.createStream();
      stream.acceptWaveform({ samples: segment.samples, sampleRate: wave.sampleRate });
      recognizer.decode(stream);
      const result: OnlineRecognizerResult = recognizer.getResult(stream);

      const text: string = `${startTime} -- ${endTime} ${result.text}`
      resultList.push(text);
      console.log(`partial result ${text}`);

      workerPort.postMessage({ 'msgType': 'non-streaming-asr-vad-decode-partial', text });

      vad.pop();
    }
  }

  return resultList.join('\n\n');
}

function decodeMic(samples: Float32Array) {
  const resultList: string[] = [];

  const windowSize: number = vad.config.sileroVad.windowSize;
  for (let i = 0; i < samples.length; i += windowSize) {
    const thisWindow: Float32Array = samples.subarray(i, i + windowSize)
    vad.acceptWaveform(thisWindow);
    if (i + windowSize >= samples.length) {
      vad.flush();
    }
    while (!vad.isEmpty()) {
      const segment: SpeechSegment = vad.front();
      const _startTime: number = (segment.start / 16000);
      const _endTime: number = _startTime + segment.samples.length / 16000;

      if (_endTime - _startTime < 0.2) {
        vad.pop();
        continue;
      }

      const startTime: string = _startTime.toFixed(2);
      const endTime: string = _endTime.toFixed(2);

      const stream: OfflineStream = recognizer.createStream();
      stream.acceptWaveform({ samples: segment.samples, sampleRate: 16000 });
      recognizer.decode(stream);
      const result: OnlineRecognizerResult = recognizer.getResult(stream);

      const text: string = `${startTime} -- ${endTime} ${result.text}`
      resultList.push(text);
      console.log(`partial result ${text}`);

      workerPort.postMessage({ 'msgType': 'non-streaming-asr-vad-mic-partial', text });

      vad.pop();
    }
  }

  return resultList.join('\n\n');
}

/**
 * Defines the event handler to be called when the worker thread receives a message sent by the host thread.
 * The event handler is executed in the worker thread.
 *
 * @param e message data
 */
workerPort.onmessage = (e: MessageEvents) => {
  const msgType = e.data['msgType'] as string;
  console.log(`msg-type: ${msgType}`)
  if (msgType == 'init-vad' && !vad) {
    const context = e.data['context'] as Context;
    vad = initVad(context);
    console.log('init vad done');
    workerPort.postMessage({ 'msgType': 'init-vad-done' });
  }

  if (msgType == 'init-vad-mic' && !vadMic) {
    const context = e.data['context'] as Context;
    vadMic = initVad(context);
    console.log('init vad mic done');
    workerPort.postMessage({ 'msgType': 'init-vad-mic-done' });
  }

  if (msgType == 'init-non-streaming-asr' && !recognizer) {
    const context = e.data['context'] as Context;
    recognizer = initNonStreamingAsr(context);
    console.log('init non streaming ASR done');
    workerPort.postMessage({ 'msgType': 'init-non-streaming-asr-done' });
  }

  if (msgType == 'non-streaming-asr-vad-decode') {
    const filename = e.data['filename'] as string;
    console.log(`decoding ${filename}`);
    try {
      const text = decodeFile(filename);
      workerPort.postMessage({ msgType: 'non-streaming-asr-vad-decode-done', text });
    } catch (e) {
      workerPort.postMessage({ msgType: 'non-streaming-asr-vad-decode-error', text: `Failed to decode ${filename}` });
    }

    workerPort.postMessage({ 'msgType': 'non-streaming-asr-vad-decode-progress', progress: 100 });
  }

  if (msgType == 'non-streaming-asr-vad-mic') {
    const samples: Float32Array = e.data['samples'] as Float32Array;
    vadMic.reset();
    try {
      const text = decodeMic(samples);
      workerPort.postMessage({ msgType: 'non-streaming-asr-vad-mic-done', text });
    } catch (e) {
      workerPort.postMessage({ msgType: 'non-streaming-asr-vad-mic-error', text: `Failed to decode` });
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
