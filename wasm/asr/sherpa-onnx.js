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

function initSherpaOnnxOnlineRecognizer() {
  let onlineTransducerModelConfig = {
    encoder: './encoder.onnx',
    decoder: './decoder.onnx',
    joiner: './joiner.onnx',
  }

  let config =
      initSherpaOnnxOnlineTransducerModelConfig(onlineTransducerModelConfig);
  _MyPrint(config.ptr);
}
