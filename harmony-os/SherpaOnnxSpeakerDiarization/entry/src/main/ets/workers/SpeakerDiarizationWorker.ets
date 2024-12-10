import worker, { ThreadWorkerGlobalScope, MessageEvents, ErrorEvent } from '@ohos.worker';
import { OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
  OfflineSpeakerDiarizationSegment,
  readWaveFromBinary, Samples } from 'sherpa_onnx';
import { fileIo } from '@kit.CoreFileKit';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

let sd: OfflineSpeakerDiarization;

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

  config.segmentation.pyannote.model = 'sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx';
  config.segmentation.numThreads = 2;
  config.segmentation.debug = true;

  config.embedding.model = '3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx';
  config.embedding.numThreads = 2;
  config.embedding.debug = true;

  config.minDurationOn = 0.2;
  config.minDurationOff = 0.5;
  return new OfflineSpeakerDiarization(config, context.resourceManager);
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
    workerPort.postMessage({msgType: 'init-speaker-diarization-done'});
    console.log('Init sd done');
  }

  if (msgType == 'speaker-diarization-file') {
    const filename = e.data['filename'] as string;
    const wave = readWave(filename);
    let result = '';
    if (wave == undefined || wave == null) {
      result = `Failed to read ${filename}`;
    } else if (wave.sampleRate != sd.sampleRate) {
      result = `Expected sample rate: ${sd.sampleRate}`;
      result += '\n';
      result += `Sample rate in file ${filename} is ${wave.sampleRate}`;
    } else {
      const duration = wave.samples.length / wave.sampleRate;
      console.log(`Processing ${filename} of ${duration} seconds`);

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
    }

    workerPort.postMessage({
      msgType: 'speaker-diarization-file-done',
      result
    });
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