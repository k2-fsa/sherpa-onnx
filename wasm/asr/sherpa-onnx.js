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


function initSherpaOnnxOnlineRecognizer() {
  let onlineTransducerModelConfig = {
    encoder: './encoder.onnx',
    decoder: './decoder.onnx',
    joiner: './joiner.onnx',
  }

  let onlineParaformerModelConfig = {
    encoder: './paraformer-encoder.onnx',
    decoder: './paraformer-decoder.onnx',
  }

  let onlineZipformer2CtcModelConfig = {
    model: './ctc.onnx',
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

  let config = initSherpaOnnxOnlineModelConfig(onlineModelConfig);

  _MyPrint(config.ptr);
  freeConfig(config)
}
