import worker, { ThreadWorkerGlobalScope, MessageEvents, ErrorEvent } from '@ohos.worker';
import { OnlineRecognizer, OnlineRecognizerConfig } from 'sherpa_onnx';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;

let recognizer: OnlineRecognizer;


function initStreamingAsr(context: Context): OnlineRecognizer {
  const config = new OnlineRecognizerConfig();
  config.modelConfig.transducer.encoder = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx';
  config.modelConfig.transducer.decoder = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx';
  config.modelConfig.transducer.joiner = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx';
  config.modelConfig.tokens = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt';
  config.modelConfig.debug = true;

  return new OnlineRecognizer(config, context.resourceManager);
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

  if (msgType == 'init-streaming-asr' && !recognizer) {
    console.log('initializing streaming ASR...');
    const context = e.data['context'] as Context;
    recognizer = initStreamingAsr(context);
    console.log('streaming ASR is initialized. ');
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