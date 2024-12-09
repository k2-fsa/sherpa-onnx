import worker, { ErrorEvent, MessageEvents, ThreadWorkerGlobalScope } from '@ohos.worker';
import {
  OnlineModelConfig,
  OnlineRecognizer,
  OnlineRecognizerConfig,
  OnlineStream,
  readWaveFromBinary,
  Samples
} from 'sherpa_onnx';
import { fileIo } from '@kit.CoreFileKit';

const workerPort: ThreadWorkerGlobalScope = worker.workerPort;


let recognizer: OnlineRecognizer;
let micStream: OnlineStream;

function getModelConfig(type: number): OnlineModelConfig {
  const modelConfig = new OnlineModelConfig();
  switch (type) {
    case 0: {
      const modelDir = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'zipformer';
      break;
    }

    case 1: {
      const modelDir = 'sherpa-onnx-lstm-zh-2023-02-20';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-11-avg-1.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-11-avg-1.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-11-avg-1.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'lstm';
      break;
    }

    case 2: {
      const modelDir = 'sherpa-onnx-lstm-en-2023-02-17';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'lstm';
      break;
    }

    case 3: {
      const modelDir = 'icefall-asr-zipformer-streaming-wenetspeech-20230615';
      modelConfig.transducer.encoder = `${modelDir}/exp/encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx`;
      modelConfig.tokens = `${modelDir}/data/lang_char/tokens.txt`;
      modelConfig.modelType = 'zipformer2';
      break;
    }

    case 4: {
      const modelDir = 'icefall-asr-zipformer-streaming-wenetspeech-20230615';
      modelConfig.transducer.encoder = `${modelDir}/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx`;
      modelConfig.tokens = `${modelDir}/data/lang_char/tokens.txt`;
      modelConfig.modelType = 'zipformer2';
      break;
    }

    case 5: {
      const modelDir = 'sherpa-onnx-streaming-paraformer-bilingual-zh-en';
      modelConfig.paraformer.encoder = `${modelDir}/encoder.int8.onnx`;
      modelConfig.paraformer.decoder = `${modelDir}/decoder.int8.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'paraformer';
      break;
    }

    case 6: {
      const modelDir = 'sherpa-onnx-streaming-zipformer-en-2023-06-26';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1-chunk-16-left-128.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1-chunk-16-left-128.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'zipformer2';
      break;
    }

    case 7: {
      const modelDir = 'sherpa-onnx-streaming-zipformer-fr-2023-04-14';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-29-avg-9-with-averaged-model.int8.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-29-avg-9-with-averaged-model.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-29-avg-9-with-averaged-model.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'zipformer';
      break;
    }

    case 8: {
      const modelDir = 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.int8.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.int8.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'zipformer';
      break;
    }

    case 9: {
      const modelDir = 'sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23'
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.int8.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.int8.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'zipformer';
      break;
    }

    case 10: {
      const modelDir = 'sherpa-onnx-streaming-zipformer-en-20M-2023-02-17';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.int8.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.int8.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'zipformer';
      break;
    }

    case 14: {
      const modelDir = 'sherpa-onnx-streaming-zipformer-korean-2024-06-16';
      modelConfig.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.int8.onnx`;
      modelConfig.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      modelConfig.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.int8.onnx`;
      modelConfig.tokens = `${modelDir}/tokens.txt`;
      modelConfig.modelType = 'zipformer';
      break;
    }
    default: {
      console.log(`Please specify a supported type. Given type ${type}`);
    }
  }
  return modelConfig;
}

function initStreamingAsr(context: Context): OnlineRecognizer {
  let type: number;

  /*

If you use type = 8, then you should have the following directory structure in the rawfile directory

(py38) fangjuns-MacBook-Pro:rawfile fangjun$ pwd
/Users/fangjun/open-source/sherpa-onnx/harmony-os/SherpaOnnxStreamingAsr/entry/src/main/resources/rawfile
(py38) fangjuns-MacBook-Pro:rawfile fangjun$ ls
sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
(py38) fangjuns-MacBook-Pro:rawfile fangjun$ tree .
.
└── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
    ├── decoder-epoch-99-avg-1.onnx
    ├── encoder-epoch-99-avg-1.int8.onnx
    ├── joiner-epoch-99-avg-1.int8.onnx
    └── tokens.txt

1 directory, 4 files

You can download model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

Note that please delete files that are not used. Otherwise, you APP will be very large
due to containing unused large files.

   */
  type = 8;

  const config: OnlineRecognizerConfig = new OnlineRecognizerConfig();
  config.modelConfig = getModelConfig(type);
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
      'msgType': 'streaming-asr-decode-file-done', text: result.text, duration: result.duration
    });
  }

  if (msgType == 'streaming-asr-decode-mic-start') {
    micStream = recognizer.createStream();
  }

  if (msgType == 'streaming-asr-decode-mic-stop') { // nothing to do
  }

  if (msgType == 'streaming-asr-decode-mic-samples') {
    const samples = e.data['samples'] as Float32Array;
    const sampleRate = e.data['sampleRate'] as number;

    micStream.acceptWaveform({ samples, sampleRate });
    while (recognizer.isReady(micStream)) {
      recognizer.decode(micStream);

      let isEndpoint = false;
      let text = recognizer.getResult(micStream).text;

      if (recognizer.isEndpoint(micStream)) {
        isEndpoint = true;
        recognizer.reset(micStream);
      }

      if (text.trim() != '') {
        workerPort.postMessage({
          'msgType': 'streaming-asr-decode-mic-result', text: text, isEndpoint: isEndpoint,
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