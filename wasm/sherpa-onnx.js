
function freeConfig(config) {
  if ('buffer' in config) {
    _free(config.buffer);
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
  let ptr = _malloc(7 * 4);

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
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += modelLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += lexiconLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += tokensLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += dataDirLen;

  Module.setValue(ptr + 16, config.noiseScale, 'float');
  Module.setValue(ptr + 20, config.noiseScaleW, 'float');
  Module.setValue(ptr + 24, config.lengthScale, 'float');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineTts() {
  let offlineTtsVitsModelConfig = initSherpaOnnxOfflineTtsVitsModelConfig({
    model: './model.onnx',
    lexicon: './lexicon.txt',
    tokens: './tokens.txt',
    dataDir: './espeak-ng-data',
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  })
  console.log(offlineTtsVitsModelConfig)
  Module._MyPrint(offlineTtsVitsModelConfig.ptr);
}
