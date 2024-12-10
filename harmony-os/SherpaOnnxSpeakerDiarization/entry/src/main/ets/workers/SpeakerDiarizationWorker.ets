import worker, { ErrorEvent, MessageEvents, ThreadWorkerGlobalScope } from '@ohos.worker';
import {
  OfflineSpeakerDiarization,
  OfflineSpeakerDiarizationConfig,
  OfflineSpeakerDiarizationSegment,
  readWaveFromBinary,
  Samples
} from 'sherpa_onnx';
import { fileIo } from '@kit.CoreFileKit';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

let sd: OfflineSpeakerDiarization;
let useAsync: boolean = true;

function readWave(filename: string): Samples {
  const fp = fileIo.openSync(filename);
  const stat = fileIo.statSync(fp.fd);
  const arrayBuffer = new ArrayBuffer(stat.size);
  fileIo.readSync(fp.fd, arrayBuffer);
  const data: Uint8Array = new Uint8Array(arrayBuffer);
  return readWaveFromBinary(data) as Samples;
}

function initOfflineSpeakerDiarization(context: Context): OfflineSpeakerDiarization {
  const config: OfflineSpeakerDiarizationConfig = new OfflineSpeakerDiarizationConfig();

  // Please refer to https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
  // to download models.
  // Make sure you have placed it inside the directory
  // harmony-os/SherpaOnnxSpeakerDiarization/entry/src/main/resources/rawfile
  //
  // Also, please delete unused files to reduce the size of the app
  config.segmentation.pyannote.model = 'sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx';
  config.segmentation.numThreads = 2;
  config.segmentation.debug = true;

  // Please refer to https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
  // to download models.
  // Make sure you have placed it inside the directory
  // harmony-os/SherpaOnnxSpeakerDiarization/entry/src/main/resources/rawfile
  config.embedding.model = '3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx';
  config.embedding.numThreads = 2;
  config.embedding.debug = true;

  config.minDurationOn = 0.2;
  config.minDurationOff = 0.5;
  return new OfflineSpeakerDiarization(config, context.resourceManager);

  // For the above two models files, you should have the following directory structure
  /*
  (py38) fangjuns-MacBook-Pro:rawfile fangjun$ pwd
  /Users/fangjun/open-source/sherpa-onnx/harmony-os/SherpaOnnxSpeakerDiarization/entry/src/main/resources/rawfile
  (py38) fangjuns-MacBook-Pro:rawfile fangjun$ ls -lh
  total 77336
  -rw-r--r--  1 fangjun  staff    38M Dec 10 16:28 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
  drwxr-xr-x  3 fangjun  staff    96B Dec 10 19:36 sherpa-onnx-pyannote-segmentation-3-0
  (py38) fangjuns-MacBook-Pro:rawfile fangjun$ tree .
  .
  ├── 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
  └── sherpa-onnx-pyannote-segmentation-3-0
      └── model.int8.onnx

  1 directory, 2 files

  (Note that we have kept only model.int8.onnx and removed all other files
  from sherpa-onnx-pyannote-segmentation-3-0
  )
   */
}

/**
 * Defines the event handler to be called when the worker thread receives a message sent by the host thread.
 * The event handler is executed in the worker thread.
 *
 * @param e message data
 */
workerPort.onmessage = (e: MessageEvents) => {
  const msgType = e.data['msgType'] as string;

  console.log(`from the main thread, msg-type: ${msgType}`);
  if (msgType == 'init-speaker-diarization' && !sd) {
    const context: Context = e.data['context'] as Context;
    sd = initOfflineSpeakerDiarization(context);
    workerPort.postMessage({ msgType: 'init-speaker-diarization-done' });
    console.log('Init sd done');
  }

  if (msgType == 'speaker-diarization-file') {
    const filename = e.data['filename'] as string;
    const numSpeakers = e.data['numSpeakers'] as number;
    const wave = readWave(filename);
    let result = '';
    if (wave == undefined || wave == null) {
      result = `Failed to read ${filename}`;

      workerPort.postMessage({
        msgType: 'speaker-diarization-file-done', result
      });
      return;
    }

    if (wave.sampleRate != sd.sampleRate) {
      result = `Expected sample rate: ${sd.sampleRate}`;
      result += '\n';
      result += `Sample rate in file ${filename} is ${wave.sampleRate}`;

      workerPort.postMessage({
        msgType: 'speaker-diarization-file-done', result
      });

      return;
    }

    const duration = wave.samples.length / wave.sampleRate;
    console.log(`Processing ${filename} of ${duration} seconds`);

    // You can remove this if statement if you want
    if (duration < 0.3) {
      result = `${filename} has only ${duration} seconds. Please use a longer file`;

      workerPort.postMessage({
        msgType: 'speaker-diarization-file-done', result
      });
      return;
    }
    sd.config.clustering.numClusters = numSpeakers;
    sd.setConfig(sd.config);

    if (useAsync) {
      sd.processAsync(wave.samples, (numProcessedChunks: number, numTotalChunks: number) => {
        const progress = numProcessedChunks / numTotalChunks * 100;
        workerPort.postMessage({
          msgType: 'speaker-diarization-file-progress', progress
        });
      }).then((r: OfflineSpeakerDiarizationSegment[]) => {
        console.log(`r is ${r.length}, ${r}`);

        for (const s of r) {
          const start: string = s.start.toFixed(3);
          const end: string = s.end.toFixed(3);
          result += `${start}\t--\t${end}\tspeaker_${s.speaker}\n`;
          console.log(`result: ${result}`);
        }

        if (r.length == 0) {
          result = 'The result is empty';
        }

        workerPort.postMessage({
          msgType: 'speaker-diarization-file-done', result
        });
      });
    } else {
      const r: OfflineSpeakerDiarizationSegment[] = sd.process(wave.samples)
      console.log(`r is ${r.length}, ${r}`);
      for (const s of r) {
        const start: string = s.start.toFixed(3);
        const end: string = s.end.toFixed(3);
        result += `${start}\t--\t${end}\tspeaker_${s.speaker}\n`;
        console.log(`result: ${result}`);
      }

      if (r.length == 0) {
        result = 'The result is empty';
      }

      workerPort.postMessage({
        msgType: 'speaker-diarization-file-done', result
      });
    }
  }
} /**
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