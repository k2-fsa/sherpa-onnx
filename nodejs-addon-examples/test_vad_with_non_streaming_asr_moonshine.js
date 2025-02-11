// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx-node');

function createRecognizer() {
  // Please download test files from
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  const config = {
    'featConfig': {
      'sampleRate': 16000,
      'featureDim': 80,
    },
    'modelConfig': {
      'moonshine': {
        'preprocessor': './sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx',
        'encoder': './sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx',
        'uncachedDecoder':
            './sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx',
        'cachedDecoder':
            './sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx',
      },
      'tokens': './sherpa-onnx-moonshine-tiny-en-int8/tokens.txt',
      'numThreads': 2,
      'provider': 'cpu',
      'debug': 1,
    }
  };

  return new sherpa_onnx.OfflineRecognizer(config);
}

function createVad() {
  // please download silero_vad.onnx from
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  const config = {
    sileroVad: {
      model: './silero_vad.onnx',
      threshold: 0.5,
      minSpeechDuration: 0.25,
      minSilenceDuration: 0.5,
      maxSpeechDuration: 5,
      windowSize: 512,
    },
    sampleRate: 16000,
    debug: true,
    numThreads: 1,
  };

  const bufferSizeInSeconds = 60;

  return new sherpa_onnx.Vad(config, bufferSizeInSeconds);
}

const recognizer = createRecognizer();
const vad = createVad();

// please download ./Obama.wav from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
const waveFilename = './Obama.wav';
const wave = sherpa_onnx.readWave(waveFilename);

if (wave.sampleRate != recognizer.config.featConfig.sampleRate) {
  throw new Error(
      'Expected sample rate: ${recognizer.config.featConfig.sampleRate}. Given: ${wave.sampleRate}');
}

console.log('Started')
let start = Date.now();

const windowSize = vad.config.sileroVad.windowSize;
for (let i = 0; i < wave.samples.length; i += windowSize) {
  const thisWindow = wave.samples.subarray(i, i + windowSize);
  vad.acceptWaveform(thisWindow);

  while (!vad.isEmpty()) {
    const segment = vad.front();
    vad.pop();

    let start_time = segment.start / wave.sampleRate;
    let end_time = start_time + segment.samples.length / wave.sampleRate;

    start_time = start_time.toFixed(2);
    end_time = end_time.toFixed(2);

    const stream = recognizer.createStream();
    stream.acceptWaveform(
        {samples: segment.samples, sampleRate: wave.sampleRate});

    recognizer.decode(stream);
    const r = recognizer.getResult(stream);
    if (r.text.length > 0) {
      const text = r.text.toLowerCase().trim();
      console.log(`${start_time} -- ${end_time}: ${text}`);
    }
  }
}

vad.flush();

while (!vad.isEmpty()) {
  const segment = vad.front();
  vad.pop();

  let start_time = segment.start / wave.sampleRate;
  let end_time = start_time + segment.samples.length / wave.sampleRate;

  start_time = start_time.toFixed(2);
  end_time = end_time.toFixed(2);

  const stream = recognizer.createStream();
  stream.acceptWaveform(
      {samples: segment.samples, sampleRate: wave.sampleRate});

  recognizer.decode(stream);
  const r = recognizer.getResult(stream);
  if (r.text.length > 0) {
    const text = r.text.toLowerCase().trim();
    console.log(`${start_time} -- ${end_time}: ${text}`);
  }
}

let stop = Date.now();
console.log('Done')

const elapsed_seconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds')
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds')
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3))
