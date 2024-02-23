function freeConfig(config) {
  if ('buffer' in config) {
    _free(config.buffer);
  }

  if ('config' in config) {
    freeConfig(config.config)
  }

  if ('transducer' in config) {
    freeConfig(config.transducer)
  }

  if ('paraformer' in config) {
    freeConfig(config.paraformer)
  }

  if ('ctc' in config) {
    freeConfig(config.ctc)
  }

  if ('feat' in config) {
    freeConfig(config.feat)
  }

  if ('model' in config) {
    freeConfig(config.model)
  }

  _free(config.ptr);
}

// The user should free the returned pointers
function initSherpaOnnxOnlineTransducerModelConfig(config) {
  let encoderLen = lengthBytesUTF8(config.encoder) + 1;
  let decoderLen = lengthBytesUTF8(config.decoder) + 1;
  let joinerLen = lengthBytesUTF8(config.joiner) + 1;

  let n = encoderLen + decoderLen + joinerLen;

  let buffer = _malloc(n);

  let len = 3 * 4;  // 3 pointers
  let ptr = _malloc(len);

  let offset = 0;
  stringToUTF8(config.encoder, buffer + offset, encoderLen);
  offset += encoderLen;

  stringToUTF8(config.decoder, buffer + offset, decoderLen);
  offset += decoderLen;

  stringToUTF8(config.joiner, buffer + offset, joinerLen);

  offset = 0;
  setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  setValue(ptr + 4, buffer + offset, 'i8*');
  offset += decoderLen;

  setValue(ptr + 8, buffer + offset, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineParaformerModelConfig(config) {
  let encoderLen = lengthBytesUTF8(config.encoder) + 1;
  let decoderLen = lengthBytesUTF8(config.decoder) + 1;

  let n = encoderLen + decoderLen;
  let buffer = _malloc(n);

  let len = 2 * 4;  // 2 pointers
  let ptr = _malloc(len);

  let offset = 0;
  stringToUTF8(config.encoder, buffer + offset, encoderLen);
  offset += encoderLen;

  stringToUTF8(config.decoder, buffer + offset, decoderLen);

  offset = 0;
  setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  setValue(ptr + 4, buffer + offset, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineZipformer2CtcModelConfig(config) {
  let n = lengthBytesUTF8(config.model) + 1;
  let buffer = _malloc(n);

  let len = 1 * 4;  // 1 pointer
  let ptr = _malloc(len);

  stringToUTF8(config.model, buffer, n);

  setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineModelConfig(config) {
  let transducer = initSherpaOnnxOnlineTransducerModelConfig(config.transducer);
  let paraformer = initSherpaOnnxOnlineParaformerModelConfig(config.paraformer);
  let ctc = initSherpaOnnxOnlineZipformer2CtcModelConfig(config.zipformer2Ctc);

  let len = transducer.len + paraformer.len + ctc.len + 5 * 4;
  let ptr = _malloc(len);

  let offset = 0;
  _CopyHeap(transducer.ptr, transducer.len, ptr + offset);
  offset += transducer.len;

  _CopyHeap(paraformer.ptr, paraformer.len, ptr + offset);
  offset += paraformer.len;

  _CopyHeap(ctc.ptr, ctc.len, ptr + offset);
  offset += ctc.len;

  let tokensLen = lengthBytesUTF8(config.tokens) + 1;
  let providerLen = lengthBytesUTF8(config.provider) + 1;
  let modelTypeLen = lengthBytesUTF8(config.modelType) + 1;
  let bufferLen = tokensLen + providerLen + modelTypeLen;
  let buffer = _malloc(bufferLen);

  offset = 0;
  stringToUTF8(config.tokens, buffer, tokensLen);
  offset += tokensLen;

  stringToUTF8(config.provider, buffer + offset, providerLen);
  offset += providerLen;

  stringToUTF8(config.modelType, buffer + offset, modelTypeLen);

  offset = transducer.len + paraformer.len + ctc.len;
  setValue(ptr + offset, buffer, 'i8*');  // tokens
  offset += 4;

  setValue(ptr + offset, config.numThreads, 'i32');
  offset += 4;

  setValue(ptr + offset, buffer + tokensLen, 'i8*');  // provider
  offset += 4;

  setValue(ptr + offset, config.debug, 'i32');
  offset += 4;

  setValue(ptr + offset, buffer + tokensLen + providerLen, 'i8*');  // modelType
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len, transducer: transducer,
        paraformer: paraformer, ctc: ctc
  }
}

function initSherpaOnnxFeatureConfig(config) {
  let len = 2 * 4;  // 2 pointers
  let ptr = _malloc(len);

  setValue(ptr, config.sampleRate, 'i32');
  setValue(ptr + 4, config.featureDim, 'i32');
  return {ptr: ptr, len: len};
}

function initSherpaOnnxOnlineRecognizerConfig(config) {
  let feat = initSherpaOnnxFeatureConfig(config.featConfig);
  let model = initSherpaOnnxOnlineModelConfig(config.modelConfig);

  let len = feat.len + model.len + 8 * 4;
  let ptr = _malloc(len);

  let offset = 0;
  _CopyHeap(feat.ptr, feat.len, ptr + offset);
  offset += feat.len;

  _CopyHeap(model.ptr, model.len, ptr + offset);
  offset += model.len;

  let decodingMethodLen = lengthBytesUTF8(config.decodingMethod) + 1;
  let hotwordsFileLen = lengthBytesUTF8(config.hotwordsFile) + 1;
  let bufferLen = decodingMethodLen + hotwordsFileLen;
  let buffer = _malloc(bufferLen);

  offset = 0;
  stringToUTF8(config.decodingMethod, buffer, decodingMethodLen);
  offset += decodingMethodLen;

  stringToUTF8(config.hotwordsFile, buffer + offset, hotwordsFileLen);

  offset = feat.len + model.len;
  setValue(ptr + offset, buffer, 'i8*');  // decoding method
  offset += 4;

  setValue(ptr + offset, config.maxActivePaths, 'i32');
  offset += 4;

  setValue(ptr + offset, config.enableEndpoint, 'i32');
  offset += 4;

  setValue(ptr + offset, config.rule1MinTrailingSilence, 'float');
  offset += 4;

  setValue(ptr + offset, config.rule2MinTrailingSilence, 'float');
  offset += 4;

  setValue(ptr + offset, config.rule3MinUtteranceLength, 'float');
  offset += 4;

  setValue(ptr + offset, buffer + decodingMethodLen, 'i8*');
  offset += 4;

  setValue(ptr + offset, config.hotwordsScore, 'float');
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len, feat: feat, model: model
  }
}


function createRecognizer() {
  let onlineTransducerModelConfig = {
    encoder: '',
    decoder: '',
    joiner: '',
  }

  let onlineParaformerModelConfig = {
    encoder: '',
    decoder: '',
  }

  let onlineZipformer2CtcModelConfig = {
    model: '',
  }

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
  }

  let featureConfig = {
    sampleRate: 16000,
    featureDim: 80,
  }

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
  }

  return new OnlineRecognizer(recognizerConfig);
}

class OnlineStream {
  constructor(handle) {
    this.handle = handle;
    this.pointer = null;  // buffer
    this.n = 0;           // buffer size
  }

  free() {
    if (this.handle) {
      _DestroyOnlineStream(this.handle);
      this.handle = null;
      _free(this.pointer);
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
      _free(this.pointer);
      this.pointer = _malloc(samples.length * samples.BYTES_PER_ELEMENT);
      this.n = samples.length
    }

    Module.HEAPF32.set(samples, this.pointer / samples.BYTES_PER_ELEMENT);
    _AcceptWaveform(this.handle, sampleRate, this.pointer, samples.length);
  }

  inputFinished() {
    _InputFinished(this.handle);
  }
};

class OnlineRecognizer {
  constructor(configObj) {
    let config = initSherpaOnnxOnlineRecognizerConfig(configObj)
    let handle = _CreateOnlineRecognizer(config.ptr);

    freeConfig(config);

    this.handle = handle;
  }

  free() {
    _DestroyOnlineRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    let handle = _CreateOnlineStream(this.handle);
    return new OnlineStream(handle);
  }

  isReady(stream) {
    return _IsOnlineStreamReady(this.handle, stream.handle) == 1;
  }

  decode(stream) {
    return _DecodeOnlineStream(this.handle, stream.handle);
  }

  isEndpoint(stream) {
    return _IsEndpoint(this.handle, stream.handle) == 1;
  }

  reset(stream) {
    _Reset(this.handle, stream.handle);
  }

  getResult(stream) {
    let r = _GetOnlineStreamResult(this.handle, stream.handle);
    let textPtr = getValue(r, 'i8*');
    let text = UTF8ToString(textPtr);
    _DestroyOnlineRecognizerResult(r);
    return text;
  }
}
