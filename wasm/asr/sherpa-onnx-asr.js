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

  if ('nemoCtc' in config) {
    freeConfig(config.nemoCtc, Module)
  }

  if ('whisper' in config) {
    freeConfig(config.whisper, Module)
  }

  if ('tdnn' in config) {
    freeConfig(config.tdnn, Module)
  }

  if ('lm' in config) {
    freeConfig(config.lm, Module)
  }

  if ('ctcFstDecoder' in config) {
    freeConfig(config.ctcFstDecoder, Module)
  }

  Module._free(config.ptr);
}

// The user should free the returned pointers
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

function initSherpaOnnxOnlineParaformerModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder) + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder) + 1;

  const n = encoderLen + decoderLen;
  const buffer = Module._malloc(n);

  const len = 2 * 4;  // 2 pointers
  const ptr = Module._malloc(len);

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
  const n = Module.lengthBytesUTF8(config.model) + 1;
  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineModelConfig(config, Module) {
  const transducer =
      initSherpaOnnxOnlineTransducerModelConfig(config.transducer, Module);
  const paraformer =
      initSherpaOnnxOnlineParaformerModelConfig(config.paraformer, Module);
  const ctc = initSherpaOnnxOnlineZipformer2CtcModelConfig(
      config.zipformer2Ctc, Module);

  const len = transducer.len + paraformer.len + ctc.len + 5 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(transducer.ptr, transducer.len, ptr + offset);
  offset += transducer.len;

  Module._CopyHeap(paraformer.ptr, paraformer.len, ptr + offset);
  offset += paraformer.len;

  Module._CopyHeap(ctc.ptr, ctc.len, ptr + offset);
  offset += ctc.len;

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
  const len = 2 * 4;  // 2 pointers
  const ptr = Module._malloc(len);

  Module.setValue(ptr, config.sampleRate, 'i32');
  Module.setValue(ptr + 4, config.featureDim, 'i32');
  return {ptr: ptr, len: len};
}

function initSherpaOnnxOnlineCtcFstDecoderConfig(config, Module) {
  const len = 2 * 4;
  const ptr = Module._malloc(len);

  const graphLen = Module.lengthBytesUTF8(config.graph) + 1;
  const buffer = Module._malloc(graphLen);
  Module.stringToUTF8(config.graph, buffer, graphLen);

  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, config.maxActive, 'i32');
  return {ptr: ptr, len: len, buffer: buffer};
}

function initSherpaOnnxOnlineRecognizerConfig(config, Module) {
  const feat = initSherpaOnnxFeatureConfig(config.featConfig, Module);
  const model = initSherpaOnnxOnlineModelConfig(config.modelConfig, Module);
  const ctcFstDecoder = initSherpaOnnxOnlineCtcFstDecoderConfig(
      config.ctcFstDecoderConfig, Module)

  const len = feat.len + model.len + 8 * 4 + ctcFstDecoder.len;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(feat.ptr, feat.len, ptr + offset);
  offset += feat.len;

  Module._CopyHeap(model.ptr, model.len, ptr + offset);
  offset += model.len;

  const decodingMethodLen = Module.lengthBytesUTF8(config.decodingMethod) + 1;
  const hotwordsFileLen = Module.lengthBytesUTF8(config.hotwordsFile) + 1;
  const bufferLen = decodingMethodLen + hotwordsFileLen;
  const buffer = Module._malloc(bufferLen);

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

  Module._CopyHeap(ctcFstDecoder.ptr, ctcFstDecoder.len, ptr + offset);

  return {
    buffer: buffer, ptr: ptr, len: len, feat: feat, model: model,
        ctcFstDecoder: ctcFstDecoder
  }
}


function createOnlineRecognizer(Module, myConfig) {
  const onlineTransducerModelConfig = {
    encoder: '',
    decoder: '',
    joiner: '',
  };

  const onlineParaformerModelConfig = {
    encoder: '',
    decoder: '',
  };

  const onlineZipformer2CtcModelConfig = {
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


  const onlineModelConfig = {
    transducer: onlineTransducerModelConfig,
    paraformer: onlineParaformerModelConfig,
    zipformer2Ctc: onlineZipformer2CtcModelConfig,
    tokens: './tokens.txt',
    numThreads: 1,
    provider: 'cpu',
    debug: 1,
    modelType: '',
  };

  const featureConfig = {
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
    ctcFstDecoderConfig: {
      graph: '',
      maxActive: 3000,
    }
  };
  if (myConfig) {
    recognizerConfig = myConfig;
  }

  return new OnlineRecognizer(recognizerConfig, Module);
}

function initSherpaOnnxOfflineTransducerModelConfig(config, Module) {
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

function initSherpaOnnxOfflineParaformerModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model) + 1;

  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineNemoEncDecCtcModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model) + 1;

  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineWhisperModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder) + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder) + 1;
  const languageLen = Module.lengthBytesUTF8(config.language) + 1;
  const taskLen = Module.lengthBytesUTF8(config.task) + 1;

  const n = encoderLen + decoderLen + languageLen + taskLen;
  const buffer = Module._malloc(n);

  const len = 4 * 4;  // 4 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder, buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder, buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.language, buffer + offset, languageLen);
  offset += languageLen;

  Module.stringToUTF8(config.task, buffer + offset, taskLen);

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += decoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += languageLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += taskLen;

  Module.setValue(ptr + 16, config.tailPaddings || -1, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineTdnnModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model) + 1;
  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model, buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineLMConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model) + 1;
  const buffer = Module._malloc(n);

  const len = 2 * 4;
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model, buffer, n);
  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, config.scale, 'float');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineModelConfig(config, Module) {
  const transducer =
      initSherpaOnnxOfflineTransducerModelConfig(config.transducer, Module);
  const paraformer =
      initSherpaOnnxOfflineParaformerModelConfig(config.paraformer, Module);
  const nemoCtc =
      initSherpaOnnxOfflineNemoEncDecCtcModelConfig(config.nemoCtc, Module);
  const whisper =
      initSherpaOnnxOfflineWhisperModelConfig(config.whisper, Module);
  const tdnn = initSherpaOnnxOfflineTdnnModelConfig(config.tdnn, Module);

  const len = transducer.len + paraformer.len + nemoCtc.len + whisper.len +
      tdnn.len + 5 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(transducer.ptr, transducer.len, ptr + offset);
  offset += transducer.len;

  Module._CopyHeap(paraformer.ptr, paraformer.len, ptr + offset);
  offset += paraformer.len;

  Module._CopyHeap(nemoCtc.ptr, nemoCtc.len, ptr + offset);
  offset += nemoCtc.len;

  Module._CopyHeap(whisper.ptr, whisper.len, ptr + offset);
  offset += whisper.len;

  Module._CopyHeap(tdnn.ptr, tdnn.len, ptr + offset);
  offset += tdnn.len;

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

  offset =
      transducer.len + paraformer.len + nemoCtc.len + whisper.len + tdnn.len;
  Module.setValue(ptr + offset, buffer, 'i8*');  // tokens
  offset += 4;

  Module.setValue(ptr + offset, config.numThreads, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.debug, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + tokensLen, 'i8*');  // provider
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + tokensLen + providerLen, 'i8*');  // modelType
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len, transducer: transducer,
        paraformer: paraformer, nemoCtc: nemoCtc, whisper: whisper, tdnn: tdnn
  }
}

function initSherpaOnnxOfflineRecognizerConfig(config, Module) {
  const feat = initSherpaOnnxFeatureConfig(config.featConfig, Module);
  const model = initSherpaOnnxOfflineModelConfig(config.modelConfig, Module);
  const lm = initSherpaOnnxOfflineLMConfig(config.lmConfig, Module);

  const len = feat.len + model.len + lm.len + 4 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(feat.ptr, feat.len, ptr + offset);
  offset += feat.len;

  Module._CopyHeap(model.ptr, model.len, ptr + offset);
  offset += model.len;

  Module._CopyHeap(lm.ptr, lm.len, ptr + offset);
  offset += lm.len;

  const decodingMethodLen = Module.lengthBytesUTF8(config.decodingMethod) + 1;
  const hotwordsFileLen = Module.lengthBytesUTF8(config.hotwordsFile) + 1;
  const bufferLen = decodingMethodLen + hotwordsFileLen;
  const buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(config.decodingMethod, buffer, decodingMethodLen);
  offset += decodingMethodLen;

  Module.stringToUTF8(config.hotwordsFile, buffer + offset, hotwordsFileLen);

  offset = feat.len + model.len + lm.len;

  Module.setValue(ptr + offset, buffer, 'i8*');  // decoding method
  offset += 4;

  Module.setValue(ptr + offset, config.maxActivePaths, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + decodingMethodLen, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.hotwordsScore, 'float');
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len, feat: feat, model: model, lm: lm
  }
}

class OfflineStream {
  constructor(handle, Module) {
    this.handle = handle;
    this.Module = Module;
  }

  free() {
    if (this.handle) {
      this.Module._DestroyOfflineStream(this.handle);
      this.handle = null;
    }
  }

  /**
   * @param sampleRate {Number}
   * @param samples {Float32Array} Containing samples in the range [-1, 1]
   */
  acceptWaveform(sampleRate, samples) {
    const pointer =
        this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
    this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
    this.Module._AcceptWaveformOffline(
        this.handle, sampleRate, pointer, samples.length);
    this.Module._free(pointer);
  }
};

class OfflineRecognizer {
  constructor(configObj, Module) {
    this.config = configObj;
    const config = initSherpaOnnxOfflineRecognizerConfig(configObj, Module);
    const handle = Module._CreateOfflineRecognizer(config.ptr);
    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._DestroyOfflineRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    const handle = this.Module._CreateOfflineStream(this.handle);
    return new OfflineStream(handle, this.Module);
  }

  decode(stream) {
    this.Module._DecodeOfflineStream(this.handle, stream.handle);
  }

  getResult(stream) {
    const r = this.Module._GetOfflineStreamResultAsJson(stream.handle);
    const jsonStr = this.Module.UTF8ToString(r);
    const ans = JSON.parse(jsonStr);
    this.Module._DestroyOfflineStreamResultJson(r);

    return ans;
  }
};

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
    const config = initSherpaOnnxOnlineRecognizerConfig(configObj, Module)
    const handle = Module._CreateOnlineRecognizer(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._DestroyOnlineRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    const handle = this.Module._CreateOnlineStream(this.handle);
    return new OnlineStream(handle, this.Module);
  }

  isReady(stream) {
    return this.Module._IsOnlineStreamReady(this.handle, stream.handle) == 1;
  }

  decode(stream) {
    this.Module._DecodeOnlineStream(this.handle, stream.handle);
  }

  isEndpoint(stream) {
    return this.Module._IsEndpoint(this.handle, stream.handle) == 1;
  }

  reset(stream) {
    this.Module._Reset(this.handle, stream.handle);
  }

  getResult(stream) {
    const r =
        this.Module._GetOnlineStreamResultAsJson(this.handle, stream.handle);
    const jsonStr = this.Module.UTF8ToString(r);
    const ans = JSON.parse(jsonStr);
    this.Module._DestroyOnlineStreamResultJson(r);

    return ans;
  }
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createOnlineRecognizer,
    OfflineRecognizer,
  };
}
