import worker, { ErrorEvent, MessageEvents, ThreadWorkerGlobalScope } from '@ohos.worker';
import {
  OnlineStream,
  readWaveFromBinary,
  Samples,
  SpeakerEmbeddingExtractor,
  SpeakerEmbeddingExtractorConfig,
  SpeakerEmbeddingManager
} from 'sherpa_onnx';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

let extractor: SpeakerEmbeddingExtractor;
let manager: SpeakerEmbeddingManager;

function readWaveFromRawfile(filename: string, context: Context): Samples {
  const data: Uint8Array = context.resourceManager.getRawFileContentSync(filename);
  return readWaveFromBinary(data) as Samples;
}

function initExtractor(context: Context): SpeakerEmbeddingExtractor {
  const config: SpeakerEmbeddingExtractorConfig = new SpeakerEmbeddingExtractorConfig();

  // Please put the model file inside the directory
  // harmony-os/SherpaOnnxSpeakerIdentification/entry/src/main/resources/rawfile
/*
(py38) fangjuns-MacBook-Pro:rawfile fangjun$ pwd
/Users/fangjun/open-source/sherpa-onnx/harmony-os/SherpaOnnxSpeakerIdentification/entry/src/main/resources/rawfile
(py38) fangjuns-MacBook-Pro:rawfile fangjun$ ls -lh
total 77336
-rw-r--r--  1 fangjun  staff    38M Dec  9 19:34 3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
 */
  // You can find more models at
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
  config.model = '3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx';
  config.numThreads = 2;
  config.debug = true;

  return new SpeakerEmbeddingExtractor(config, context.resourceManager);
}

function extractEmbedding(samples: Samples): Float32Array {
  const stream: OnlineStream = extractor.createStream();
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

    workerPort.postMessage({
      msgType: 'manager-all-speaker-names', allSpeakers: manager.getAllSpeakerNames(),
    });
  }

  if (msgType == 'manager-delete-speaker') {
    const name = e.data['name'] as string;
    const ok: boolean = manager.remove(name);
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

  if (msgType == 'manager-add-speaker') {
    const name = e.data['name'] as string;
    const samples = e.data['samples'] as Float32Array;
    const sampleRate = e.data['sampleRate'] as number;

    const v = extractEmbedding({ samples, sampleRate });
    const ok: boolean = manager.add({ name, v });
    if (ok) {
      workerPort.postMessage({
        msgType: 'manager-add-speaker-done',
        status: `Added ${name}`,
        ok,
      });
      workerPort.postMessage({
        msgType: 'manager-all-speaker-names', allSpeakers: manager.getAllSpeakerNames(),
      }
      );
    } else {
      workerPort.postMessage({
        msgType: 'manager-add-speaker-done',
        status: `Failed to add ${name}. Possibly due to exsiting speaker name. Please recheck`,
        ok,
      });
    }
  }

  if (msgType == 'manager-search-speaker') {
    const threshold = e.data['threshold'] as number;
    const samples = e.data['samples'] as Float32Array;
    const sampleRate = e.data['sampleRate'] as number;

    const v = extractEmbedding({ samples, sampleRate });
    let name: string = manager.search({ threshold, v });
    if (name == '' || name == undefined) {
      name = "===<Unknown>===";
    }
    workerPort.postMessage({
      msgType: 'manager-search-speaker-done',
      name
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
