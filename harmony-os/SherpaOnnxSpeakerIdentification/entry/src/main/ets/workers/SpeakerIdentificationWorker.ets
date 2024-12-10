import worker, { ThreadWorkerGlobalScope, MessageEvents, ErrorEvent } from '@ohos.worker';
import {
  readWaveFromBinary,
  Samples,
  SpeakerEmbeddingExtractor,
  SpeakerEmbeddingExtractorConfig,
  SpeakerEmbeddingManager
} from 'sherpa_onnx';
import { fileIo } from '@kit.CoreFileKit';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

let extractor: SpeakerEmbeddingExtractor;
let manager: SpeakerEmbeddingManager;

function readWaveFromRawfile(filename: string, context: Context): Samples {
  const data: Uint8Array = context.resourceManager.getRawFileContentSync(filename);
  return readWaveFromBinary(data) as Samples;
}

function initExtractor(context: Context): SpeakerEmbeddingExtractor {
  const config = new SpeakerEmbeddingExtractorConfig();
  config.model = '3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx';
  config.numThreads = 2;
  config.debug = true;

  return new SpeakerEmbeddingExtractor(config, context.resourceManager);
}

function extractEmbedding(samples: Samples): Float32Array {
  const stream = extractor.createStream();
  stream.acceptWaveform(samples);
  return extractor.compute(stream);
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

  if (msgType == 'init-extractor' && !extractor) {
    const context: Context = e.data['context'] as Context;
    extractor = initExtractor(context);
    manager = new SpeakerEmbeddingManager(extractor.dim);

    const filename1 = 'sr-data/enroll/fangjun-sr-1.wav';
    const samples1 = readWaveFromRawfile(filename1, context);
    console.log(`sample rate: ${samples1.sampleRate}`);
    let ok = manager.add({ name: 'fangjun0', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun1', v: extractEmbedding(samples1) });
    /*
    ok = manager.add({ name: 'fangjun2', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun3', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun4', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun5', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun6', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun7', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun8', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun9', v: extractEmbedding(samples1) });
    ok = manager.add({ name: 'fangjun10', v: extractEmbedding(samples1) });
    */

    if (ok) {
      console.log(`Added fangjun`);
      let n = manager.getNumSpeakers();
      console.log(`number of speakers: ${n}`);
      console.log(`speaker names: ${manager.getAllSpeakerNames().join('\n')}`);
    }

    workerPort.postMessage({
      msgType: 'manager-all-speaker-names', allSpeakers: manager.getAllSpeakerNames(),
    });
  }

  if (msgType == 'manager-delete-speaker') {
    const name = e.data['name'] as string;
    const ok = manager.remove(name);
    if (ok) {
      console.log(`Removed ${name}.`);

      console.log(`Number of speakers: ${manager.getNumSpeakers()}`);
      console.log(`Number of speakers2: ${manager.getAllSpeakerNames().length}`);
      console.log(JSON.stringify(manager.getAllSpeakerNames()));
      workerPort.postMessage({
        msgType: 'manager-all-speaker-names', allSpeakers: manager.getAllSpeakerNames(),
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