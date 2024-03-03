function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('config' in config) {
    freeConfig(config.config, Module)
  }

  if ('transducer' in config) {
    freeConfig(config.transducer, Module)
  }

  if ('paraformer' in config) {
    freeConfig(config.paraformer, Module)
  }

  if ('ctc' in config) {
    freeConfig(config.ctc, Module)
  }

  if ('feat' in config) {
    freeConfig(config.feat, Module)
  }

  if ('model' in config) {
    freeConfig(config.model, Module)
  }

  Module._free(config.ptr);
}

// The user should free the returned pointers
function initSherpaOnnxOnlineTransducerModelConfig(config, Module) {
  let encoderLen = Module.lengthBytesUTF8(config.encoder) + 1;
  let decoderLen = Module.lengthBytesUTF8(config.decoder) + 1;
  let joinerLen = Module.lengthBytesUTF8(config.joiner) + 1;

  let n = encoderLen + decoderLen + joinerLen;

  let buffer = Module._malloc(n);

  let len = 3 * 4;  // 3 pointers
  let ptr = Module._malloc(len);

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

function initSherpaOnnxOnlineParaformerModelConfig(config, Module) {
  let encoderLen = Module.lengthBytesUTF8(config.encoder) + 1;
  let decoderLen = Module.lengthBytesUTF8(config.decoder) + 1;

  let n = encoderLen + decoderLen;
  let buffer = Module._malloc(n);

  let len = 2 * 4;  // 2 pointers
  let ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder, buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder, buffer + offset, decoderLen);

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineZipformer2CtcModelConfig(config, Module) {
  let n = Module.lengthBytesUTF8(config.model) + 1;
  let buffer = Module._malloc(n);

  let len = 1 * 4;  // 1 pointer
  let ptr = Module._malloc(len);

  Module.stringToUTF8(config.model, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineModelConfig(config, Module) {
  let transducer =
      initSherpaOnnxOnlineTransducerModelConfig(config.transducer, Module);
  let paraformer =
      initSherpaOnnxOnlineParaformerModelConfig(config.paraformer, Module);
  let ctc = initSherpaOnnxOnlineZipformer2CtcModelConfig(
      config.zipformer2Ctc, Module);

  let len = transducer.len + paraformer.len + ctc.len + 5 * 4;
  let ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(transducer.ptr, transducer.len, ptr + offset);
  offset += transducer.len;

  Module._CopyHeap(paraformer.ptr, paraformer.len, ptr + offset);
  offset += paraformer.len;

  Module._CopyHeap(ctc.ptr, ctc.len, ptr + offset);
  offset += ctc.len;

  let tokensLen = Module.lengthBytesUTF8(config.tokens) + 1;
  let providerLen = Module.lengthBytesUTF8(config.provider) + 1;
  let modelTypeLen = Module.lengthBytesUTF8(config.modelType) + 1;
  let bufferLen = tokensLen + providerLen + modelTypeLen;
  let buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(config.tokens, buffer, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.provider, buffer + offset, providerLen);
  offset += providerLen;

  Module.stringToUTF8(config.modelType, buffer + offset, modelTypeLen);

  offset = transducer.len + paraformer.len + ctc.len;
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
    buffer: buffer, ptr: ptr, len: len, transducer: transducer,
        paraformer: paraformer, ctc: ctc
  }
}

function initSherpaOnnxFeatureConfig(config, Module) {
  let len = 2 * 4;  // 2 pointers
  let ptr = Module._malloc(len);

  Module.setValue(ptr, config.sampleRate, 'i32');
  Module.setValue(ptr + 4, config.featureDim, 'i32');
  return {ptr: ptr, len: len};
}

function initSherpaOnnxOnlineRecognizerConfig(config, Module) {
  let feat = initSherpaOnnxFeatureConfig(config.featConfig, Module);
  let model = initSherpaOnnxOnlineModelConfig(config.modelConfig, Module);

  let len = feat.len + model.len + 8 * 4;
  let ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(feat.ptr, feat.len, ptr + offset);
  offset += feat.len;

  Module._CopyHeap(model.ptr, model.len, ptr + offset);
  offset += model.len;

  let decodingMethodLen = Module.lengthBytesUTF8(config.decodingMethod) + 1;
  let hotwordsFileLen = Module.lengthBytesUTF8(config.hotwordsFile) + 1;
  let bufferLen = decodingMethodLen + hotwordsFileLen;
  let buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(config.decodingMethod, buffer, decodingMethodLen);
  offset += decodingMethodLen;

  Module.stringToUTF8(config.hotwordsFile, buffer + offset, hotwordsFileLen);

  offset = feat.len + model.len;
  Module.setValue(ptr + offset, buffer, 'i8*');  // decoding method
  offset += 4;

  Module.setValue(ptr + offset, config.maxActivePaths, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.enableEndpoint, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.rule1MinTrailingSilence, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.rule2MinTrailingSilence, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.rule3MinUtteranceLength, 'float');
  offset += 4;

  Module.setValue(ptr + offset, buffer + decodingMethodLen, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.hotwordsScore, 'float');
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len, feat: feat, model: model
  }
}


function createOnlineRecognizer(Module, myConfig) {
  let onlineTransducerModelConfig = {
    encoder: '',
    decoder: '',
    joiner: '',
  };

  let onlineParaformerModelConfig = {
    encoder: '',
    decoder: '',
  };

  let onlineZipformer2CtcModelConfig = {
    model: '',
  };

  let type = 0;

  switch (type) {
    case 0:
      // transducer
      onlineTransducerModelConfig.encoder = './encoder.onnx';
      onlineTransducerModelConfig.decoder = './decoder.onnx';
      onlineTransducerModelConfig.joiner = './joiner.onnx';
      break;
    case 1:
      // paraformer
      onlineParaformerModelConfig.encoder = './encoder.onnx';
      onlineParaformerModelConfig.decoder = './decoder.onnx';
      break;
    case 2:
      // ctc
      onlineZipformer2CtcModelConfig.model = './encoder.onnx';
      break;
  }


  let onlineModelConfig = {
    transducer: onlineTransducerModelConfig,
    paraformer: onlineParaformerModelConfig,
    zipformer2Ctc: onlineZipformer2CtcModelConfig,
    tokens: './tokens.txt',
    numThreads: 1,
    provider: 'cpu',
    debug: 1,
    modelType: '',
  };

  let featureConfig = {
    sampleRate: 16000,
    featureDim: 80,
  };

  let recognizerConfig = {
    featConfig: featureConfig,
    modelConfig: onlineModelConfig,
    decodingMethod: 'greedy_search',
    maxActivePaths: 4,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 2.4,
    rule2MinTrailingSilence: 1.2,
    rule3MinUtteranceLength: 20,
    hotwordsFile: '',
    hotwordsScore: 1.5,
  };
  if (myConfig) {
    recognizerConfig = myConfig;
  }

  return new OnlineRecognizer(recognizerConfig, Module);
}

class OnlineStream {
  constructor(handle, Module) {
    this.handle = handle;
    this.pointer = null;  // buffer
    this.n = 0;           // buffer size
    this.Module = Module;
  }

  free() {
    if (this.handle) {
      this.Module._DestroyOnlineStream(this.handle);
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

class OnlineRecognizer {
  constructor(configObj, Module) {
    this.config = configObj;
    let config = initSherpaOnnxOnlineRecognizerConfig(configObj, Module)
    let handle = Module._CreateOnlineRecognizer(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._DestroyOnlineRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    let handle = this.Module._CreateOnlineStream(this.handle);
    return new OnlineStream(handle, this.Module);
  }

  isReady(stream) {
    return this.Module._IsOnlineStreamReady(this.handle, stream.handle) == 1;
  }

  decode(stream) {
    return this.Module._DecodeOnlineStream(this.handle, stream.handle);
  }

  isEndpoint(stream) {
    return this.Module._IsEndpoint(this.handle, stream.handle) == 1;
  }

  reset(stream) {
    this.Module._Reset(this.handle, stream.handle);
  }

  getResult(stream) {
    console.log('here1');
    let r = this.Module._GetOnlineStreamResult(this.handle, stream.handle);
    console.log('here2');
    let textPtr = this.Module.getValue(r, 'i8*');
    console.log('here3');
    let text = this.Module.UTF8ToString(textPtr);
    console.log('here4');
    this.Module._DestroyOnlineRecognizerResult(r);
    console.log('here5');
    return text;
  }
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createOnlineRecognizer,
  };
}
