
function freeConfig(config) {
  if ('buffer' in config) {
    _free(config.buffer);
  }

  if ('config' in config) {
    freeConfig(config.config)
  }

  _free(config.ptr);
}

// The user should free the returned pointers
function initSherpaOnnxOfflineTtsVitsModelConfig(config) {
  let modelLen = lengthBytesUTF8(config.model) + 1;
  let lexiconLen = lengthBytesUTF8(config.lexicon) + 1;
  let tokensLen = lengthBytesUTF8(config.tokens) + 1;
  let dataDirLen = lengthBytesUTF8(config.dataDir) + 1;

  let n = modelLen + lexiconLen + tokensLen + dataDirLen;

  let buffer = _malloc(n);

  let len = 7 * 4;
  let ptr = _malloc(len);

  let offset = 0;
  stringToUTF8(config.model, buffer + offset, modelLen);
  offset += modelLen;

  stringToUTF8(config.lexicon, buffer + offset, lexiconLen);
  offset += lexiconLen;

  stringToUTF8(config.tokens, buffer + offset, tokensLen);
  offset += tokensLen;

  stringToUTF8(config.dataDir, buffer + offset, dataDirLen);
  offset += dataDirLen;

  offset = 0;
  setValue(ptr, buffer + offset, 'i8*');
  offset += modelLen;

  setValue(ptr + 4, buffer + offset, 'i8*');
  offset += lexiconLen;

  setValue(ptr + 8, buffer + offset, 'i8*');
  offset += tokensLen;

  setValue(ptr + 12, buffer + offset, 'i8*');
  offset += dataDirLen;

  setValue(ptr + 16, config.noiseScale, 'float');
  setValue(ptr + 20, config.noiseScaleW, 'float');
  setValue(ptr + 24, config.lengthScale, 'float');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineTtsModelConfig(config) {
  let vitsModelConfig =
      initSherpaOnnxOfflineTtsVitsModelConfig(config.offlineTtsVitsModelConfig);

  let len = vitsModelConfig.len + 3 * 4;
  let ptr = _malloc(len);

  let offset = 0;
  _CopyHeap(vitsModelConfig.ptr, vitsModelConfig.len, ptr + offset);
  offset += vitsModelConfig.len;

  setValue(ptr + offset, config.numThreads, 'i32');
  offset += 4;

  setValue(ptr + offset, config.debug, 'i32');
  offset += 4;

  let providerLen = lengthBytesUTF8(config.provider) + 1;
  let buffer = _malloc(providerLen);
  stringToUTF8(config.provider, buffer, providerLen);
  setValue(ptr + offset, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len, config: vitsModelConfig,
  }
}

function initSherpaOnnxOfflineTtsConfig(config) {
  let modelConfig =
      initSherpaOnnxOfflineTtsModelConfig(config.offlineTtsModelConfig);
  let len = modelConfig.len + 2 * 4;
  let ptr = _malloc(len);

  let offset = 0;
  _CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset);
  offset += modelConfig.len;

  let ruleFstsLen = lengthBytesUTF8(config.ruleFsts) + 1;
  let buffer = _malloc(ruleFstsLen);
  stringToUTF8(config.ruleFsts, buffer, ruleFstsLen);
  setValue(ptr + offset, buffer, 'i8*');
  offset += 4;

  setValue(ptr + offset, config.maxNumSentences, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: len, config: modelConfig,
  }
}

class OfflineTts {
  constructor(configObj) {
    let config = initSherpaOnnxOfflineTtsConfig(configObj)
    let handle = _SherpaOnnxCreateOfflineTts(config.ptr);

    freeConfig(config);

    this.handle = handle;
    this.sampleRate = _SherpaOnnxOfflineTtsSampleRate(this.handle);
    this.numSpeakers = _SherpaOnnxOfflineTtsNumSpeakers(this.handle);
  }

  free() {
    _SherpaOnnxDestroyOfflineTts(this.handle);
    this.handle = 0
  }

  // {
  //   text: "hello",
  //   sid: 1,
  //   speed: 1.0
  // }
  generate(config) {
    let textLen = lengthBytesUTF8(config.text) + 1;
    let textPtr = _malloc(textLen);
    stringToUTF8(config.text, textPtr, textLen);

    let h = _SherpaOnnxOfflineTtsGenerate(
        this.handle, textPtr, config.sid, config.speed);

    let numSamples = HEAP32[h / 4 + 1];
    let sampleRate = HEAP32[h / 4 + 2];

    let samplesPtr = HEAP32[h / 4] / 4;
    let samples = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      samples[i] = HEAPF32[samplesPtr + i];
    }

    _SherpaOnnxDestroyOfflineTtsGeneratedAudio(h);
    return {samples: samples, sampleRate: sampleRate};
  }
}

function initSherpaOnnxOfflineTts() {
  let offlineTtsVitsModelConfig = {
    model: './model.onnx',
    lexicon: '',
    tokens: './tokens.txt',
    dataDir: './espeak-ng-data',
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };
  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    ruleFsts: '',
    maxNumSentences: 1,
  }

  return new OfflineTts(offlineTtsConfig);
}
