import worker, { ThreadWorkerGlobalScope, MessageEvents, ErrorEvent } from '@ohos.worker';
import { OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, readWaveFromBinary, Samples } from 'sherpa_onnx';
import { fileIo } from '@kit.CoreFileKit';
import { audio } from '@kit.AudioKit';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;


let recognizer: OnlineRecognizer;
let micStream: OnlineStream;


function initStreamingAsr(context: Context): OnlineRecognizer {
  const config: OnlineRecognizerConfig = new OnlineRecognizerConfig();
  config.modelConfig.transducer.encoder =
    'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx';
  config.modelConfig.transducer.decoder =
    'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx';
  config.modelConfig.transducer.joiner =
    'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx';
  config.modelConfig.tokens = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt';
  config.modelConfig.debug = true;
  config.modelConfig.numThreads = 2;

  config.enableEndpoint = true;

  return new OnlineRecognizer(config, context.resourceManager);
}

interface DecodeFileResult {
  text: string;
  duration: number;
}

function decodeFile(filename: string): DecodeFileResult {
  const fp = fileIo.openSync(filename);
  const stat = fileIo.statSync(fp.fd);
  const arrayBuffer = new ArrayBuffer(stat.size);
  fileIo.readSync(fp.fd, arrayBuffer);
  const data: Uint8Array = new Uint8Array(arrayBuffer);
  const wave: Samples = readWaveFromBinary(data) as Samples;
  console.log(`Sample rate: ${wave.sampleRate}`);

  const stream = recognizer.createStream();
  stream.acceptWaveform(wave);
  const tailPadding = new Float32Array(0.5 * wave.sampleRate);
  tailPadding.fill(0);

  stream.acceptWaveform({ samples: tailPadding, sampleRate: wave.sampleRate });

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  const audioDuration = wave.samples.length / wave.sampleRate;

  return { text: recognizer.getResult(stream).text, duration: audioDuration };
}

/**
 * Defines the event handler to be called when the worker thread receives a message sent by the host thread.
 * The event handler is executed in the worker thread.
 *
 * @param e message data
 */
workerPort.onmessage = (e: MessageEvents) => {
  const msgType = e.data['msgType'] as string;

  if (msgType != 'streaming-asr-decode-mic-samples') {
    console.log(`from the main thread, msg-type: ${msgType}`);
  }

  if (msgType == 'init-streaming-asr' && !recognizer) {
    console.log('initializing streaming ASR...');
    const context = e.data['context'] as Context;
    recognizer = initStreamingAsr(context);
    console.log('streaming ASR is initialized. ');
    workerPort.postMessage({ 'msgType': 'init-streaming-asr-done' });
  }

  if (msgType == 'streaming-asr-decode-file') {
    const filename = e.data['filename'] as string;
    console.log(`decoding ${filename}`);
    const result = decodeFile(filename);
    workerPort.postMessage({
      'msgType': 'streaming-asr-decode-file-done',
      text: result.text,
      duration: result.duration
    });
  }

  if (msgType == 'streaming-asr-decode-mic-start') {
    micStream = recognizer.createStream();
  }

  if (msgType == 'streaming-asr-decode-mic-stop') {
    // nothing to do
  }

  if (msgType == 'streaming-asr-decode-mic-samples') {
    const samples = e.data['samples'] as Float32Array;
    const sampleRate = e.data['sampleRate'] as number;

    micStream.acceptWaveform({samples, sampleRate});
    while (recognizer.isReady(micStream)) {
      recognizer.decode(micStream);

      let isEndpoint = false;
      let text = recognizer.getResult(micStream).text;

      if (recognizer.isEndpoint(micStream)) {
        isEndpoint = true;
        recognizer.reset(micStream);
      }

      console.log(`text: ${text}, isEndpoint: ${isEndpoint}`);

      if (text.trim() != '') {
        workerPort.postMessage({
          'msgType': 'streaming-asr-decode-mic-result',
          text: text,
          isEndpoint: isEndpoint,
        });
      }
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