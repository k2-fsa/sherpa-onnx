

function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }
  Module._free(config.ptr);
}

// The user should free the returned pointers
function initModelConfig(config, Module) {

  let encoderBinLen = Module.lengthBytesUTF8(config.transducer.encoder) + 1;
  let decoderBinLen = Module.lengthBytesUTF8(config.transducer.decoder) + 1;
  let joinerBinLen = Module.lengthBytesUTF8(config.transducer.joiner) + 1;

  let tokensLen = Module.lengthBytesUTF8(config.tokens) + 1;

  let n = encoderBinLen + decoderBinLen + joinerBinLen + tokensLen;

  let buffer = Module._malloc(n);
  let ptr = Module._malloc(4 * 5);

  let offset = 0;
  Module.stringToUTF8(config.transducer.encoder, buffer + offset, encoderBinLen);
  offset += encoderBinLen;

  Module.stringToUTF8(config.transducer.decoder, buffer + offset, decoderBinLen);
  offset += encoderBinLen;

  Module.stringToUTF8(config.transducer.joiner, buffer + offset, joinerBinLen);
  offset += joinerBinLen;

  Module.stringToUTF8(config.tokens, buffer + offset, tokensLen);
  offset += tokensLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');  // encoderBin
  offset += encoderBinLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');  // decoderBin
  offset += decoderBinLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');  // joinerBin
  offset += joinerBinLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');  // tokens
  offset += tokensLen;

  Module.setValue(ptr + 16, config.numThreads, 'i32');  // numThread

  return {
    buffer: buffer, ptr: ptr, len: 20,
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
  let featConfig =
      initFeatureExtractorConfig(config.featConfig, Module);

  let modelConfig = initModelConfig(config.modelConfig, Module);
  let numBytes =
      featConfig.len + modelConfig.len + 4 * 5;

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
    ptr: ptr, len: numBytes, featConfig: featConfig, modelConfig: modelConfig
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
    _InputFinished(this.handle);
  }
};

class Kws {
  constructor(configObj, Module) {
    this.config = configObj;
    let config = initKwsConfig(configObj, Module)
    let handle = Module._CreateOnlineKws(config.ptr);


    freeConfig(config.featConfig, Module);
    freeConfig(config.modelConfig, Module);
    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._DestroyOnlineKws(this.handle);
    this.handle = 0
  }

  createStream() {
    let handle = this.Module._CreateOnlineKwsStream(this.handle);
    return new Stream(handle, this.Module);
  }

  isReady(stream) {
    return this.Module._IsOnlineKwsStreamReady(this.handle, stream.handle) === 1;
  }


  decode(stream) {
    return this.Module._DecodeOnlineKwsStream(this.handle, stream.handle);
  }

  getResult(stream) {
    let r = this.Module._GetOnlineKwsStreamResult(this.handle, stream.handle);
    let jsonPtr = this.Module.getValue(r + 16, 'i8*');
    let json = this.Module.UTF8ToString(jsonPtr);
    this.Module._DestroyOnlineKwsResult(r);
    return JSON.parse(json);
  }
}

function createKws(Module, myConfig) {
  let transducerConfig = {
    encoder: './encoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    decoder: './decoder-epoch-12-avg-2-chunk-16-left-64.onnx',
    joiner: './joiner-epoch-12-avg-2-chunk-16-left-64.onnx',
  }
  let modelConfig = {
    transducer: transducerConfig,
    tokens: './tokens.txt',
    numThreads: 1
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
    keywords: "x iǎo ài t óng x ué @小爱同学\n" +
        "j ūn g ē n iú b ī @军哥牛逼"
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