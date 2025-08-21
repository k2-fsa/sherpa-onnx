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

  if ('zipformer2Ctc' in config) {
    freeConfig(config.zipformer2Ctc, Module)
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

  if ('fireRedAsr' in config) {
    freeConfig(config.fireRedAsr, Module)
  }

  if ('dolphin' in config) {
    freeConfig(config.dolphin, Module)
  }

  if ('zipformerCtc' in config) {
    freeConfig(config.zipformerCtc, Module)
  }

  if ('moonshine' in config) {
    freeConfig(config.moonshine, Module)
  }

  if ('tdnn' in config) {
    freeConfig(config.tdnn, Module)
  }

  if ('senseVoice' in config) {
    freeConfig(config.senseVoice, Module)
  }

  if ('canary' in config) {
    freeConfig(config.canary, Module)
  }

  if ('lm' in config) {
    freeConfig(config.lm, Module)
  }

  if ('ctcFstDecoder' in config) {
    freeConfig(config.ctcFstDecoder, Module)
  }

  if ('hr' in config) {
    freeConfig(config.hr, Module)
  }

  Module._free(config.ptr);
}

// The user should free the returned pointers
function initSherpaOnnxOnlineTransducerModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;
  const joinerLen = Module.lengthBytesUTF8(config.joiner || '') + 1;

  const n = encoderLen + decoderLen + joinerLen;

  const buffer = Module._malloc(n);

  const len = 3 * 4;  // 3 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.joiner || '', buffer + offset, joinerLen);

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
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;

  const n = encoderLen + decoderLen;
  const buffer = Module._malloc(n);

  const len = 2 * 4;  // 2 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineZipformer2CtcModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model || '') + 1;
  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineNemoCtcModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model || '') + 1;
  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOnlineModelConfig(config, Module) {
  if (!('transducer' in config)) {
    config.transducer = {
      encoder: '',
      decoder: '',
      joiner: '',
    };
  }

  if (!('paraformer' in config)) {
    config.paraformer = {
      encoder: '',
      decoder: '',
    };
  }

  if (!('zipformer2Ctc' in config)) {
    config.zipformer2Ctc = {
      model: '',
    };
  }

  if (!('nemoCtc' in config)) {
    config.nemoCtc = {
      model: '',
    };
  }

  if (!('tokensBuf' in config)) {
    config.tokensBuf = '';
  }

  if (!('tokensBufSize' in config)) {
    config.tokensBufSize = 0;
  }

  const transducer =
      initSherpaOnnxOnlineTransducerModelConfig(config.transducer, Module);

  const paraformer =
      initSherpaOnnxOnlineParaformerModelConfig(config.paraformer, Module);

  const zipformer2Ctc = initSherpaOnnxOnlineZipformer2CtcModelConfig(
      config.zipformer2Ctc, Module);

  const nemoCtc =
      initSherpaOnnxOnlineNemoCtcModelConfig(config.nemoCtc, Module);

  const len =
      transducer.len + paraformer.len + zipformer2Ctc.len + 9 * 4 + nemoCtc.len;

  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(transducer.ptr, transducer.len, ptr + offset);
  offset += transducer.len;

  Module._CopyHeap(paraformer.ptr, paraformer.len, ptr + offset);
  offset += paraformer.len;

  Module._CopyHeap(zipformer2Ctc.ptr, zipformer2Ctc.len, ptr + offset);
  offset += zipformer2Ctc.len;

  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const modelTypeLen = Module.lengthBytesUTF8(config.modelType || '') + 1;
  const modelingUnitLen = Module.lengthBytesUTF8(config.modelingUnit || '') + 1;
  const bpeVocabLen = Module.lengthBytesUTF8(config.bpeVocab || '') + 1;
  const tokensBufLen = Module.lengthBytesUTF8(config.tokensBuf || '') + 1;

  const bufferLen = tokensLen + providerLen + modelTypeLen + modelingUnitLen +
      bpeVocabLen + tokensBufLen;
  const buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(config.tokens || '', buffer, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.provider || 'cpu', buffer + offset, providerLen);
  offset += providerLen;

  Module.stringToUTF8(config.modelType || '', buffer + offset, modelTypeLen);
  offset += modelTypeLen;

  Module.stringToUTF8(
      config.modelingUnit || '', buffer + offset, modelingUnitLen);
  offset += modelingUnitLen;

  Module.stringToUTF8(config.bpeVocab || '', buffer + offset, bpeVocabLen);
  offset += bpeVocabLen;

  Module.stringToUTF8(config.tokensBuf || '', buffer + offset, tokensBufLen);
  offset += tokensBufLen;

  offset = transducer.len + paraformer.len + zipformer2Ctc.len;
  Module.setValue(ptr + offset, buffer, 'i8*');  // tokens
  offset += 4;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + tokensLen, 'i8*');  // provider
  offset += 4;

  Module.setValue(ptr + offset, config.debug ?? 1, 'i32');
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + tokensLen + providerLen, 'i8*');  // modelType
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + tokensLen + providerLen + modelTypeLen,
      'i8*');  // modelingUnit
  offset += 4;

  Module.setValue(
      ptr + offset,
      buffer + tokensLen + providerLen + modelTypeLen + modelingUnitLen,
      'i8*');  // bpeVocab
  offset += 4;

  Module.setValue(
      ptr + offset,
      buffer + tokensLen + providerLen + modelTypeLen + modelingUnitLen +
          bpeVocabLen,
      'i8*');  // tokens_buf
  offset += 4;

  Module.setValue(ptr + offset, config.tokensBufSize || 0, 'i32');
  offset += 4;

  Module._CopyHeap(nemoCtc.ptr, nemoCtc.len, ptr + offset);
  offset += nemoCtc.len;

  return {
    buffer: buffer, ptr: ptr, len: len, transducer: transducer,
        paraformer: paraformer, zipformer2Ctc: zipformer2Ctc, nemoCtc: nemoCtc
  }
}

function initSherpaOnnxFeatureConfig(config, Module) {
  const len = 2 * 4;  // 2 pointers
  const ptr = Module._malloc(len);

  Module.setValue(ptr, config.sampleRate || 16000, 'i32');
  Module.setValue(ptr + 4, config.featureDim || 80, 'i32');
  return {ptr: ptr, len: len};
}

function initSherpaOnnxHomophoneReplacerConfig(config, Module) {
  const len = 3 * 4;
  const ptr = Module._malloc(len);

  const dictDirLen = Module.lengthBytesUTF8(config.dictDir || '') + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
  const ruleFstsLen = Module.lengthBytesUTF8(config.ruleFsts || '') + 1;

  const bufferLen = dictDirLen + lexiconLen + ruleFstsLen;

  const buffer = Module._malloc(bufferLen);
  let offset = 0
  Module.stringToUTF8(config.dictDir || '', buffer + offset, dictDirLen);
  offset += dictDirLen;

  Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
  offset += lexiconLen;

  Module.stringToUTF8(config.ruleFsts || '', buffer + offset, ruleFstsLen);
  offset += ruleFstsLen;

  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, buffer + dictDirLen, 'i8*');
  Module.setValue(ptr + 8, buffer + dictDirLen + lexiconLen, 'i8*');

  return {ptr: ptr, len: len, buffer: buffer};
}

function initSherpaOnnxOnlineCtcFstDecoderConfig(config, Module) {
  const len = 2 * 4;
  const ptr = Module._malloc(len);

  const graphLen = Module.lengthBytesUTF8(config.graph || '') + 1;
  const buffer = Module._malloc(graphLen);
  Module.stringToUTF8(config.graph, buffer, graphLen);

  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, config.maxActive || 3000, 'i32');
  return {ptr: ptr, len: len, buffer: buffer};
}

function initSherpaOnnxOnlineRecognizerConfig(config, Module) {
  if (!('featConfig' in config)) {
    config.featConfig = {
      sampleRate: 16000,
      featureDim: 80,
    };
  }

  if (!('ctcFstDecoderConfig' in config)) {
    config.ctcFstDecoderConfig = {
      graph: '',
      maxActive: 3000,
    };
  }

  if (!('hotwordsBuf' in config)) {
    config.hotwordsBuf = '';
  }

  if (!('hotwordsBufSize' in config)) {
    config.hotwordsBufSize = 0;
  }

  if (!('hr' in config)) {
    config.hr = {
      dictDir: '',
      lexicon: '',
      ruleFsts: '',
    };
  }

  const feat = initSherpaOnnxFeatureConfig(config.featConfig, Module);
  const model = initSherpaOnnxOnlineModelConfig(config.modelConfig, Module);
  const ctcFstDecoder = initSherpaOnnxOnlineCtcFstDecoderConfig(
      config.ctcFstDecoderConfig, Module)
  const hr = initSherpaOnnxHomophoneReplacerConfig(config.hr, Module);

  const len = feat.len + model.len + 8 * 4 + ctcFstDecoder.len + 5 * 4 + hr.len;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(feat.ptr, feat.len, ptr + offset);
  offset += feat.len;

  Module._CopyHeap(model.ptr, model.len, ptr + offset);
  offset += model.len;

  const decodingMethodLen =
      Module.lengthBytesUTF8(config.decodingMethod || 'greedy_search') + 1;
  const hotwordsFileLen = Module.lengthBytesUTF8(config.hotwordsFile || '') + 1;
  const ruleFstsFileLen = Module.lengthBytesUTF8(config.ruleFsts || '') + 1;
  const ruleFarsFileLen = Module.lengthBytesUTF8(config.ruleFars || '') + 1;
  const hotwordsBufLen = Module.lengthBytesUTF8(config.hotwordsBuf || '') + 1;
  const bufferLen = decodingMethodLen + hotwordsFileLen + ruleFstsFileLen +
      ruleFarsFileLen + hotwordsBufLen;
  const buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(
      config.decodingMethod || 'greedy_search', buffer, decodingMethodLen);
  offset += decodingMethodLen;

  Module.stringToUTF8(
      config.hotwordsFile || '', buffer + offset, hotwordsFileLen);
  offset += hotwordsFileLen;

  Module.stringToUTF8(config.ruleFsts || '', buffer + offset, ruleFstsFileLen);
  offset += ruleFstsFileLen;

  Module.stringToUTF8(config.ruleFars || '', buffer + offset, ruleFarsFileLen);
  offset += ruleFarsFileLen;

  Module.stringToUTF8(
      config.hotwordsBuf || '', buffer + offset, hotwordsBufLen);
  offset += hotwordsBufLen;

  offset = feat.len + model.len;
  Module.setValue(ptr + offset, buffer, 'i8*');  // decoding method
  offset += 4;

  Module.setValue(ptr + offset, config.maxActivePaths || 4, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.enableEndpoint || 0, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.rule1MinTrailingSilence || 2.4, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.rule2MinTrailingSilence || 1.2, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.rule3MinUtteranceLength || 20, 'float');
  offset += 4;

  Module.setValue(ptr + offset, buffer + decodingMethodLen, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.hotwordsScore || 1.5, 'float');
  offset += 4;

  Module._CopyHeap(ctcFstDecoder.ptr, ctcFstDecoder.len, ptr + offset);
  offset += ctcFstDecoder.len;

  Module.setValue(
      ptr + offset, buffer + decodingMethodLen + hotwordsFileLen, 'i8*');
  offset += 4;

  Module.setValue(
      ptr + offset,
      buffer + decodingMethodLen + hotwordsFileLen + ruleFstsFileLen, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.blankPenalty || 0, 'float');
  offset += 4;

  Module.setValue(
      ptr + offset,
      buffer + decodingMethodLen + hotwordsFileLen + ruleFstsFileLen +
          ruleFarsFileLen,
      'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.hotwordsBufSize || 0, 'i32');
  offset += 4;

  Module._CopyHeap(hr.ptr, hr.len, ptr + offset);
  offset += hr.len;

  return {
    buffer: buffer, ptr: ptr, len: len, feat: feat, model: model,
        ctcFstDecoder: ctcFstDecoder, hr: hr,
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

  const onlineNemoCtcModelConfig = {
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
      // zipformer2Ctc
      onlineZipformer2CtcModelConfig.model = './encoder.onnx';
      break;
    case 3:
      // nemoCtc
      onlineNemoCtcModelConfig.model = './nemo-ctc.onnx';
      break;
  }


  const onlineModelConfig = {
    transducer: onlineTransducerModelConfig,
    paraformer: onlineParaformerModelConfig,
    zipformer2Ctc: onlineZipformer2CtcModelConfig,
    nemoCtc: onlineNemoCtcModelConfig,
    tokens: './tokens.txt',
    numThreads: 1,
    provider: 'cpu',
    debug: 1,
    modelType: '',
    modelingUnit: 'cjkchar',
    bpeVocab: '',
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
    },
    ruleFsts: '',
    ruleFars: '',
  };
  if (myConfig) {
    recognizerConfig = myConfig;
  }

  return new OnlineRecognizer(recognizerConfig, Module);
}

function initSherpaOnnxOfflineTransducerModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;
  const joinerLen = Module.lengthBytesUTF8(config.joiner || '') + 1;

  const n = encoderLen + decoderLen + joinerLen;

  const buffer = Module._malloc(n);

  const len = 3 * 4;  // 3 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.joiner || '', buffer + offset, joinerLen);

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
  const n = Module.lengthBytesUTF8(config.model || '') + 1;

  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineNemoEncDecCtcModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model || '') + 1;

  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineDolphinModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model || '') + 1;

  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineZipformerCtcModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model || '') + 1;

  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineWhisperModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;
  const languageLen = Module.lengthBytesUTF8(config.language || '') + 1;
  const taskLen = Module.lengthBytesUTF8(config.task || '') + 1;

  const n = encoderLen + decoderLen + languageLen + taskLen;
  const buffer = Module._malloc(n);

  const len = 5 * 4;  // 4 pointers + 1 int32
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.language || '', buffer + offset, languageLen);
  offset += languageLen;

  Module.stringToUTF8(config.task || '', buffer + offset, taskLen);

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += decoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += languageLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += taskLen;

  Module.setValue(ptr + 16, config.tailPaddings || 2000, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineCanaryModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;
  const srcLangLen = Module.lengthBytesUTF8(config.srcLang || '') + 1;
  const tgtLangLen = Module.lengthBytesUTF8(config.tgtLang || '') + 1;

  const n = encoderLen + decoderLen + srcLangLen + tgtLangLen;
  const buffer = Module._malloc(n);

  const len = 5 * 4;  // 4 pointers + 1 int32
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.srcLang || '', buffer + offset, srcLangLen);
  offset += srcLangLen;

  Module.stringToUTF8(config.tgtLang || '', buffer + offset, tgtLangLen);
  offset += tgtLangLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += decoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += srcLangLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += tgtLangLen;

  Module.setValue(ptr + 16, config.usePnc ?? 1, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineMoonshineModelConfig(config, Module) {
  const preprocessorLen = Module.lengthBytesUTF8(config.preprocessor || '') + 1;
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const uncachedDecoderLen =
      Module.lengthBytesUTF8(config.uncachedDecoder || '') + 1;
  const cachedDecoderLen =
      Module.lengthBytesUTF8(config.cachedDecoder || '') + 1;

  const n =
      preprocessorLen + encoderLen + uncachedDecoderLen + cachedDecoderLen;
  const buffer = Module._malloc(n);

  const len = 4 * 4;  // 4 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(
      config.preprocessor || '', buffer + offset, preprocessorLen);
  offset += preprocessorLen;

  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(
      config.uncachedDecoder || '', buffer + offset, uncachedDecoderLen);
  offset += uncachedDecoderLen;

  Module.stringToUTF8(
      config.cachedDecoder || '', buffer + offset, cachedDecoderLen);
  offset += cachedDecoderLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += preprocessorLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += uncachedDecoderLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += cachedDecoderLen;

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineFireRedAsrModelConfig(config, Module) {
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;

  const n = encoderLen + decoderLen;
  const buffer = Module._malloc(n);

  const len = 2 * 4;  // 2 pointers
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);
  offset += decoderLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += decoderLen;

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineTdnnModelConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model || '') + 1;
  const buffer = Module._malloc(n);

  const len = 1 * 4;  // 1 pointer
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);

  Module.setValue(ptr, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineSenseVoiceModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;
  const languageLen = Module.lengthBytesUTF8(config.language || '') + 1;

  // useItn is a integer with 4 bytes
  const n = modelLen + languageLen;
  const buffer = Module._malloc(n);

  const len = 3 * 4;  // 2 pointers + 1 int
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
  offset += modelLen;

  Module.stringToUTF8(config.language || '', buffer + offset, languageLen);
  offset += languageLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += modelLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += languageLen;

  Module.setValue(ptr + 8, config.useInverseTextNormalization ?? 0, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineLMConfig(config, Module) {
  const n = Module.lengthBytesUTF8(config.model || '') + 1;
  const buffer = Module._malloc(n);

  const len = 2 * 4;
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, n);
  Module.setValue(ptr, buffer, 'i8*');
  Module.setValue(ptr + 4, config.scale || 1, 'float');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineModelConfig(config, Module) {
  if (!('transducer' in config)) {
    config.transducer = {
      encoder: '',
      decoder: '',
      joiner: '',
    };
  }

  if (!('paraformer' in config)) {
    config.paraformer = {
      model: '',
    };
  }

  if (!('nemoCtc' in config)) {
    config.nemoCtc = {
      model: '',
    };
  }

  if (!('dolphin' in config)) {
    config.dolphin = {
      model: '',
    };
  }

  if (!('zipformerCtc' in config)) {
    config.zipformerCtc = {
      model: '',
    };
  }

  if (!('whisper' in config)) {
    config.whisper = {
      encoder: '',
      decoder: '',
      language: '',
      task: '',
      tailPaddings: -1,
    };
  }

  if (!('moonshine' in config)) {
    config.moonshine = {
      preprocessor: '',
      encoder: '',
      uncachedDecoder: '',
      cachedDecoder: '',
    };
  }

  if (!('fireRedAsr' in config)) {
    config.fireRedAsr = {
      encoder: '',
      decoder: '',
    };
  }

  if (!('tdnn' in config)) {
    config.tdnn = {
      model: '',
    };
  }

  if (!('senseVoice' in config)) {
    config.senseVoice = {
      model: '',
      language: '',
      useInverseTextNormalization: 0,
    };
  }

  if (!('canary' in config)) {
    config.canary = {
      encoder: '',
      decoder: '',
      srcLang: '',
      tgtLang: '',
      usePnc: 1,
    };
  }

  const transducer =
      initSherpaOnnxOfflineTransducerModelConfig(config.transducer, Module);

  const paraformer =
      initSherpaOnnxOfflineParaformerModelConfig(config.paraformer, Module);

  const nemoCtc =
      initSherpaOnnxOfflineNemoEncDecCtcModelConfig(config.nemoCtc, Module);

  const whisper =
      initSherpaOnnxOfflineWhisperModelConfig(config.whisper, Module);

  const tdnn = initSherpaOnnxOfflineTdnnModelConfig(config.tdnn, Module);

  const senseVoice =
      initSherpaOnnxOfflineSenseVoiceModelConfig(config.senseVoice, Module);

  const moonshine =
      initSherpaOnnxOfflineMoonshineModelConfig(config.moonshine, Module);

  const fireRedAsr =
      initSherpaOnnxOfflineFireRedAsrModelConfig(config.fireRedAsr, Module);

  const dolphin =
      initSherpaOnnxOfflineDolphinModelConfig(config.dolphin, Module);

  const zipformerCtc =
      initSherpaOnnxOfflineZipformerCtcModelConfig(config.zipformerCtc, Module);

  const canary = initSherpaOnnxOfflineCanaryModelConfig(config.canary, Module);

  const len = transducer.len + paraformer.len + nemoCtc.len + whisper.len +
      tdnn.len + 8 * 4 + senseVoice.len + moonshine.len + fireRedAsr.len +
      dolphin.len + zipformerCtc.len + canary.len;

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

  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const modelTypeLen = Module.lengthBytesUTF8(config.modelType || '') + 1;
  const modelingUnitLen = Module.lengthBytesUTF8(config.modelingUnit || '') + 1;
  const bpeVocabLen = Module.lengthBytesUTF8(config.bpeVocab || '') + 1;
  const teleSpeechCtcLen =
      Module.lengthBytesUTF8(config.teleSpeechCtc || '') + 1;

  const bufferLen = tokensLen + providerLen + modelTypeLen + modelingUnitLen +
      bpeVocabLen + teleSpeechCtcLen;

  const buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(config.tokens, buffer, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.provider || 'cpu', buffer + offset, providerLen);
  offset += providerLen;

  Module.stringToUTF8(config.modelType || '', buffer + offset, modelTypeLen);
  offset += modelTypeLen;

  Module.stringToUTF8(
      config.modelingUnit || '', buffer + offset, modelingUnitLen);
  offset += modelingUnitLen;

  Module.stringToUTF8(config.bpeVocab || '', buffer + offset, bpeVocabLen);
  offset += bpeVocabLen;

  Module.stringToUTF8(
      config.teleSpeechCtc || '', buffer + offset, teleSpeechCtcLen);
  offset += teleSpeechCtcLen;

  offset =
      transducer.len + paraformer.len + nemoCtc.len + whisper.len + tdnn.len;
  Module.setValue(ptr + offset, buffer, 'i8*');  // tokens
  offset += 4;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.debug ?? 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + tokensLen, 'i8*');  // provider
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + tokensLen + providerLen, 'i8*');  // modelType
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + tokensLen + providerLen + modelTypeLen,
      'i8*');  // modelingUnit
  offset += 4;

  Module.setValue(
      ptr + offset,
      buffer + tokensLen + providerLen + modelTypeLen + modelingUnitLen,
      'i8*');  // bpeVocab
  offset += 4;

  Module.setValue(
      ptr + offset,
      buffer + tokensLen + providerLen + modelTypeLen + modelingUnitLen +
          bpeVocabLen,
      'i8*');  // teleSpeechCtc
  offset += 4;

  Module._CopyHeap(senseVoice.ptr, senseVoice.len, ptr + offset);
  offset += senseVoice.len;

  Module._CopyHeap(moonshine.ptr, moonshine.len, ptr + offset);
  offset += moonshine.len;

  Module._CopyHeap(fireRedAsr.ptr, fireRedAsr.len, ptr + offset);
  offset += fireRedAsr.len;

  Module._CopyHeap(dolphin.ptr, dolphin.len, ptr + offset);
  offset += dolphin.len;

  Module._CopyHeap(zipformerCtc.ptr, zipformerCtc.len, ptr + offset);
  offset += zipformerCtc.len;

  Module._CopyHeap(canary.ptr, canary.len, ptr + offset);
  offset += canary.len;

  return {
    buffer: buffer, ptr: ptr, len: len, transducer: transducer,
        paraformer: paraformer, nemoCtc: nemoCtc, whisper: whisper, tdnn: tdnn,
        senseVoice: senseVoice, moonshine: moonshine, fireRedAsr: fireRedAsr,
        dolphin: dolphin, zipformerCtc: zipformerCtc, canary: canary,
  }
}

function initSherpaOnnxOfflineRecognizerConfig(config, Module) {
  if (!('featConfig' in config)) {
    config.featConfig = {
      sampleRate: 16000,
      featureDim: 80,
    };
  }

  if (!('lmConfig' in config)) {
    config.lmConfig = {
      model: '',
      scale: 1.0,
    };
  }

  if (!('hr' in config)) {
    config.hr = {
      dictDir: '',
      lexicon: '',
      ruleFsts: '',
    };
  }

  const feat = initSherpaOnnxFeatureConfig(config.featConfig, Module);
  const model = initSherpaOnnxOfflineModelConfig(config.modelConfig, Module);
  const lm = initSherpaOnnxOfflineLMConfig(config.lmConfig, Module);
  const hr = initSherpaOnnxHomophoneReplacerConfig(config.hr, Module);

  const len = feat.len + model.len + lm.len + 7 * 4 + hr.len;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(feat.ptr, feat.len, ptr + offset);
  offset += feat.len;

  Module._CopyHeap(model.ptr, model.len, ptr + offset);
  offset += model.len;

  Module._CopyHeap(lm.ptr, lm.len, ptr + offset);
  offset += lm.len;

  const decodingMethodLen =
      Module.lengthBytesUTF8(config.decodingMethod || 'greedy_search') + 1;
  const hotwordsFileLen = Module.lengthBytesUTF8(config.hotwordsFile || '') + 1;
  const ruleFstsLen = Module.lengthBytesUTF8(config.ruleFsts || '') + 1;
  const ruleFarsLen = Module.lengthBytesUTF8(config.ruleFars || '') + 1;
  const bufferLen =
      decodingMethodLen + hotwordsFileLen + ruleFstsLen + ruleFarsLen;
  const buffer = Module._malloc(bufferLen);

  offset = 0;
  Module.stringToUTF8(
      config.decodingMethod || 'greedy_search', buffer, decodingMethodLen);
  offset += decodingMethodLen;

  Module.stringToUTF8(
      config.hotwordsFile || '', buffer + offset, hotwordsFileLen);
  offset += hotwordsFileLen;

  Module.stringToUTF8(config.ruleFsts || '', buffer + offset, ruleFstsLen);
  offset += ruleFstsLen;

  Module.stringToUTF8(config.ruleFars || '', buffer + offset, ruleFarsLen);
  offset += ruleFarsLen;

  offset = feat.len + model.len + lm.len;

  Module.setValue(ptr + offset, buffer, 'i8*');  // decoding method
  offset += 4;

  Module.setValue(ptr + offset, config.maxActivePaths || 4, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + decodingMethodLen, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.hotwordsScore || 1.5, 'float');
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + decodingMethodLen + hotwordsFileLen, 'i8*');
  offset += 4;

  Module.setValue(
      ptr + offset, buffer + decodingMethodLen + hotwordsFileLen + ruleFstsLen,
      'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.blankPenalty || 0, 'float');
  offset += 4;

  Module._CopyHeap(hr.ptr, hr.len, ptr + offset);
  offset += hr.len;

  return {
    buffer: buffer, ptr: ptr, len: len, feat: feat, model: model, lm: lm,
        hr: hr,
  }
}

class OfflineStream {
  constructor(handle, Module) {
    this.handle = handle;
    this.Module = Module;
  }

  free() {
    if (this.handle) {
      this.Module._SherpaOnnxDestroyOfflineStream(this.handle);
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
    this.Module._SherpaOnnxAcceptWaveformOffline(
        this.handle, sampleRate, pointer, samples.length);
    this.Module._free(pointer);
  }
};

class OfflineRecognizer {
  constructor(configObj, Module) {
    this.config = configObj;
    const config = initSherpaOnnxOfflineRecognizerConfig(configObj, Module);
    const handle = Module._SherpaOnnxCreateOfflineRecognizer(config.ptr);
    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  setConfig(configObj) {
    const config =
        initSherpaOnnxOfflineRecognizerConfig(configObj, this.Module);
    this.Module._SherpaOnnxOfflineRecognizerSetConfig(this.handle, config.ptr);
    freeConfig(config, this.Module);
  }

  free() {
    this.Module._SherpaOnnxDestroyOfflineRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    const handle = this.Module._SherpaOnnxCreateOfflineStream(this.handle);
    return new OfflineStream(handle, this.Module);
  }

  decode(stream) {
    this.Module._SherpaOnnxDecodeOfflineStream(this.handle, stream.handle);
  }

  getResult(stream) {
    const r =
        this.Module._SherpaOnnxGetOfflineStreamResultAsJson(stream.handle);
    const jsonStr = this.Module.UTF8ToString(r);
    const ans = JSON.parse(jsonStr);
    this.Module._SherpaOnnxDestroyOfflineStreamResultJson(r);

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
      this.Module._SherpaOnnxDestroyOnlineStream(this.handle);
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
    this.Module._SherpaOnnxOnlineStreamAcceptWaveform(
        this.handle, sampleRate, this.pointer, samples.length);
  }

  inputFinished() {
    this.Module._SherpaOnnxOnlineStreamInputFinished(this.handle);
  }
};

class OnlineRecognizer {
  constructor(configObj, Module) {
    this.config = configObj;
    const config = initSherpaOnnxOnlineRecognizerConfig(configObj, Module)
    const handle = Module._SherpaOnnxCreateOnlineRecognizer(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._SherpaOnnxDestroyOnlineRecognizer(this.handle);
    this.handle = 0
  }

  createStream() {
    const handle = this.Module._SherpaOnnxCreateOnlineStream(this.handle);
    return new OnlineStream(handle, this.Module);
  }

  isReady(stream) {
    return this.Module._SherpaOnnxIsOnlineStreamReady(
               this.handle, stream.handle) == 1;
  }

  decode(stream) {
    this.Module._SherpaOnnxDecodeOnlineStream(this.handle, stream.handle);
  }

  isEndpoint(stream) {
    return this.Module._SherpaOnnxOnlineStreamIsEndpoint(
               this.handle, stream.handle) == 1;
  }

  reset(stream) {
    this.Module._SherpaOnnxOnlineStreamReset(this.handle, stream.handle);
  }

  getResult(stream) {
    const r = this.Module._SherpaOnnxGetOnlineStreamResultAsJson(
        this.handle, stream.handle);
    const jsonStr = this.Module.UTF8ToString(r);
    const ans = JSON.parse(jsonStr);
    this.Module._SherpaOnnxDestroyOnlineStreamResultJson(r);

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
