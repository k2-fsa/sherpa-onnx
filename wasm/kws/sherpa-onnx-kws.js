

function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('transducer' in config) {
    freeConfig(config.transducer, Module);
  }

  if ('featConfig' in config) {
    freeConfig(config.featConfig, Module);
  }

  if ('modelConfig' in config) {
    freeConfig(config.modelConfig, Module);
  }

  if ('keywordsBuffer' in config) {
    Module._free(config.keywordsBuffer);
  }

  Module._free(config.ptr);
}


function initSherpaOnnxOnlineTransducerModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder) + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder) + 1;
  const joinerLen = Module.lengthBytesUTF8(config.joiner) + 1;

  const n = encoderLen + decoderLen + joinerLen;

  const buffer = Module._malloc(n);

  const len = 3 * 4;  // 3 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder, buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder, buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.joiner, buffer + offset, joinerLen);

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += decoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

// The user should free the returned pointers
function initModelConfig(config, Module) {
  const transducer =
      initSherpaOnnxOnlineTransducerModelConfig(config.transducer, Module);
  const paraformer_len = 2 * 4
  const ctc_len = 1 * 4

  const len = transducer.len + paraformer_len + ctc_len + 5 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(transducer.ptr, transducer.len, ptr + offset);

  const tokensLen = Module.lengthBytesUTF8(config.tokens) + 1;
  const providerLen = Module.lengthBytesUTF8(config.provider) + 1;
  const modelTypeLen = Module.lengthBytesUTF8(config.modelType) + 1;
  const bufferLen = tokensLen + providerLen + modelTypeLen;
  const buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(config.tokens, buffer, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.provider, buffer + offset, providerLen);
  offset += providerLen;

  Module.stringToUTF8(config.modelType, buffer + offset, modelTypeLen);

  offset = transducer.len + paraformer_len + ctc_len;
  Module.setValue(ptr + offset, buffer, 'i8*');  // tokens
  offset += 4;

  Module.setValue(ptr + offset, config.numThreads, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + tokensLen, 'i8*');  // provider
  offset += 4;

  Module.setValue(ptr + offset, config.debug, 'i32');
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + tokensLen + providerLen, 'i8*');  // modelType
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len, transducer: transducer
  }
}

function initFeatureExtractorConfig(config, Module) {
  let ptr = Module._malloc(4 * 2);
  Module.setValue(ptr, config.samplingRate, 'i32');
  Module.setValue(ptr + 4, config.featureDim, 'i32');
  return {
    ptr: ptr, len: 8,
  }
}

function initKwsConfig(config, Module) {
  let featConfig = initFeatureExtractorConfig(config.featConfig, Module);

  let modelConfig = initModelConfig(config.modelConfig, Module);
  let numBytes = featConfig.len + modelConfig.len + 4 * 5;

  let ptr = Module._malloc(numBytes);
  let offset = 0;
  Module._CopyHeap(featConfig.ptr, featConfig.len, ptr + offset);
  offset += featConfig.len;

  Module._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset)
  offset += modelConfig.len;

  Module.setValue(ptr + offset, config.maxActivePaths, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.numTrailingBlanks, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.keywordsScore, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.keywordsThreshold, 'float');
  offset += 4;

  let keywordsLen = Module.lengthBytesUTF8(config.keywords) + 1;
  let keywordsBuffer = Module._malloc(keywordsLen);
  Module.stringToUTF8(config.keywords, keywordsBuffer, keywordsLen);
  Module.setValue(ptr + offset, keywordsBuffer, 'i8*');
  offset += 4;

  return {
    ptr: ptr, len: numBytes, featConfig: featConfig, modelConfig: modelConfig,
        keywordsBuffer: keywordsBuffer
  }
}

class Stream {
  constructor(handle, Module) {
    this.handle = handle;
    this.pointer = null;
    this.n = 0;
    this.Module = Module;
  }

  free() {
    if (this.handle) {
      this.Module._DestroyOnlineKwsStream(this.handle);
      this.handle = null;
      this.Module._free(this.pointer);
      this.pointer = null;
      this.n = 0;
    }
  }

  /**
   * @param sampleRate {Number}
   * @param samples {Float32Array} Containing samples in the range [-1, 1]
   */
  acceptWaveform(sampleRate, samples) {
    if (this.n < samples.length) {
      this.Module._free(this.pointer);
      this.pointer =
          this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
      this.n = samples.length
    }

    this.Module.HEAPF32.set(samples, this.pointer / samples.BYTES_PER_ELEMENT);
    this.Module._AcceptWaveform(
        this.handle, sampleRate, this.pointer, samples.length);
  }

  inputFinished() {
    this.Module._InputFinished(this.handle);
  }
};

class Kws {
  constructor(configObj, Module) {
    this.config = configObj;
    let config = initKwsConfig(configObj, Module)
    let handle = Module._CreateKeywordSpotter(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._DestroyKeywordSpotter(this.handle);
    this.handle = 0
  }

  createStream() {
    let handle = this.Module._CreateKeywordStream(this.handle);
    return new Stream(handle, this.Module);
  }

  isReady(stream) {
    return this.Module._IsKeywordStreamReady(this.handle, stream.handle) === 1;
  }

  decode(stream) {
    return this.Module._DecodeKeywordStream(this.handle, stream.handle);
  }

  getResult(stream) {
    let r = this.Module._GetKeywordResult(this.handle, stream.handle);
    let jsonPtr = this.Module.getValue(r + 24, 'i8*');
    let json = this.Module.UTF8ToString(jsonPtr);
    this.Module._DestroyKeywordResult(r);
    return JSON.parse(json);
  }
}

function createKws(Module, myConfig) {
  let transducerConfig = {
    encoder: './encoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    decoder: './decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    joiner: './joiner-epoch-12-avg-2-chunk-16-left-64.onnx',
  };
  let modelConfig = {
    transducer: transducerConfig,
    tokens: './tokens.txt',
    provider: 'cpu',
    modelType: '',
    numThreads: 1,
    debug: 1
  };

  let featConfig = {
    samplingRate: 16000,
    featureDim: 80,
  };

  let configObj = {
    featConfig: featConfig,
    modelConfig: modelConfig,
    maxActivePaths: 4,
    numTrailingBlanks: 1,
    keywordsScore: 1.0,
    keywordsThreshold: 0.25,
    keywords: 'x iǎo ài t óng x ué @小爱同学\n' +
        'j ūn g ē n iú b ī @军哥牛逼'
  };

  if (myConfig) {
    configObj = myConfig;
  }
  return new Kws(configObj, Module);
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createKws,
  };
}
