
function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('config' in config) {
    freeConfig(config.config, Module)
  }

  if ('matcha' in config) {
    freeConfig(config.matcha, Module)
  }

  if ('kokoro' in config) {
    freeConfig(config.kokoro, Module)
  }

  if ('kitten' in config) {
    freeConfig(config.kitten, Module)
  }

  if ('zipvoice' in config) {
    freeConfig(config.zipvoice, Module)
  }

  if ('pocket' in config) {
    freeConfig(config.pocket, Module)
  }

  if ('supertonic' in config) {
    freeConfig(config.supertonic, Module)
  }

  if (config.ptr) {
    Module._free(config.ptr);
  }
}

// The user should free the returned pointers
//
// Note on `dataDir` (all model configs below, vits/matcha/kokoro/kitten/
// zipvoice): deprecated and ignored since v2.0.0. espeak-ng is no longer used
// by sherpa-onnx. The slot is still marshalled for ABI stability, but the C
// side discards it. Use `lexicon` for plain text, or pass pre-phonemized
// input via `phonemeCodepoints` / `tokens` in the generation config.
function initSherpaOnnxOfflineTtsVitsModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
  const dictDir = '';
  const dictDirLen = Module.lengthBytesUTF8(dictDir) + 1;

  const n = modelLen + lexiconLen + tokensLen + dataDirLen + dictDirLen;

  const buffer = Module._malloc(n);

  const len = 8 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
  offset += modelLen;

  Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
  offset += lexiconLen;

  Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
  offset += dataDirLen;

  Module.stringToUTF8(dictDir, buffer + offset, dictDirLen);
  offset += dictDirLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += modelLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += lexiconLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += tokensLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += dataDirLen;

  Module.setValue(ptr + 16, config.noiseScale || 0.667, 'float');
  Module.setValue(ptr + 20, config.noiseScaleW || 0.8, 'float');
  Module.setValue(ptr + 24, config.lengthScale || 1.0, 'float');
  Module.setValue(ptr + 28, buffer + offset, 'i8*');
  offset += dictDirLen;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsMatchaModelConfig(config, Module) {
  const acousticModelLen = Module.lengthBytesUTF8(config.acousticModel) + 1;
  const vocoderLen = Module.lengthBytesUTF8(config.vocoder) + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;

  const dictDir = '';
  const dictDirLen = Module.lengthBytesUTF8(dictDir) + 1;

  const n = acousticModelLen + vocoderLen + lexiconLen + tokensLen +
      dataDirLen + dictDirLen;

  const buffer = Module._malloc(n);

  const len = 8 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(
      config.acousticModel || '', buffer + offset, acousticModelLen);
  offset += acousticModelLen;

  Module.stringToUTF8(config.vocoder || '', buffer + offset, vocoderLen);
  offset += vocoderLen;

  Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
  offset += lexiconLen;

  Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
  offset += dataDirLen;

  Module.stringToUTF8(dictDir, buffer + offset, dictDirLen);
  offset += dictDirLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += acousticModelLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += vocoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += lexiconLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += tokensLen;

  Module.setValue(ptr + 16, buffer + offset, 'i8*');
  offset += dataDirLen;

  Module.setValue(ptr + 20, config.noiseScale || 0.667, 'float');
  Module.setValue(ptr + 24, config.lengthScale || 1.0, 'float');
  Module.setValue(ptr + 28, buffer + offset, 'i8*');
  offset += dictDirLen;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsKokoroModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model) + 1;
  const voicesLen = Module.lengthBytesUTF8(config.voices) + 1;
  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
  const dictDir = '';
  const dictDirLen = Module.lengthBytesUTF8(dictDir) + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
  const langLen = Module.lengthBytesUTF8(config.lang || '') + 1;

  const n = modelLen + voicesLen + tokensLen + dataDirLen + dictDirLen +
      lexiconLen + langLen;

  const buffer = Module._malloc(n);

  const len = 8 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
  offset += modelLen;

  Module.stringToUTF8(config.voices || '', buffer + offset, voicesLen);
  offset += voicesLen;

  Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
  offset += dataDirLen;

  Module.stringToUTF8(dictDir, buffer + offset, dictDirLen);
  offset += dictDirLen;

  Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
  offset += lexiconLen;

  Module.stringToUTF8(config.lang || '', buffer + offset, langLen);
  offset += langLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += modelLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += voicesLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += tokensLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += dataDirLen;

  Module.setValue(ptr + 16, config.lengthScale || 1.0, 'float');

  Module.setValue(ptr + 20, buffer + offset, 'i8*');
  offset += dictDirLen;

  Module.setValue(ptr + 24, buffer + offset, 'i8*');
  offset += lexiconLen;

  Module.setValue(ptr + 28, buffer + offset, 'i8*');
  offset += langLen;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsKittenModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model) + 1;
  const voicesLen = Module.lengthBytesUTF8(config.voices) + 1;
  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;

  const n = modelLen + voicesLen + tokensLen + dataDirLen + lexiconLen;

  const buffer = Module._malloc(n);

  const len = 6 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
  offset += modelLen;

  Module.stringToUTF8(config.voices || '', buffer + offset, voicesLen);
  offset += voicesLen;

  Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
  offset += dataDirLen;

  Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
  offset += lexiconLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += modelLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += voicesLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += tokensLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += dataDirLen;

  Module.setValue(ptr + 16, config.lengthScale || 1.0, 'float');

  Module.setValue(ptr + 20, buffer + offset, 'i8*');

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsZipVoiceModelConfig(config, Module) {
  // Support Dart camelCase: targetRms -> targetRMS
  if ('targetRms' in config && !('targetRMS' in config)) {
    config.targetRMS = config.targetRms;
  }

  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;
  const vocoderLen = Module.lengthBytesUTF8(config.vocoder || '') + 1;
  const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;

  const n = tokensLen + encoderLen + decoderLen + vocoderLen + dataDirLen +
      lexiconLen;

  const buffer = Module._malloc(n);

  const len = 10 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.tokens || '', buffer + offset, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(config.vocoder || '', buffer + offset, vocoderLen);
  offset += vocoderLen;

  Module.stringToUTF8(config.dataDir || '', buffer + offset, dataDirLen);
  offset += dataDirLen;

  Module.stringToUTF8(config.lexicon || '', buffer + offset, lexiconLen);
  offset += lexiconLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += tokensLen;

  Module.setValue(ptr + 4, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 8, buffer + offset, 'i8*');
  offset += decoderLen;

  Module.setValue(ptr + 12, buffer + offset, 'i8*');
  offset += vocoderLen;

  Module.setValue(ptr + 16, buffer + offset, 'i8*');
  offset += dataDirLen;

  Module.setValue(ptr + 20, buffer + offset, 'i8*');
  offset += lexiconLen;

  Module.setValue(ptr + 24, config.featScale || 0.1, 'float');
  Module.setValue(ptr + 28, config.tShift || 0.5, 'float');
  Module.setValue(ptr + 32, config.targetRMS || 0.1, 'float');
  Module.setValue(ptr + 36, config.guidanceScale || 1.0, 'float');

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsPocketModelConfig(config, Module) {
  const lmFlowLen = Module.lengthBytesUTF8(config.lmFlow || '') + 1;
  const lmMainLen = Module.lengthBytesUTF8(config.lmMain || '') + 1;
  const encoderLen = Module.lengthBytesUTF8(config.encoder || '') + 1;
  const decoderLen = Module.lengthBytesUTF8(config.decoder || '') + 1;
  const textConditionerLen =
      Module.lengthBytesUTF8(config.textConditioner || '') + 1;
  const vocabJsonLen = Module.lengthBytesUTF8(config.vocabJson || '') + 1;
  const tokenScoresJsonLen =
      Module.lengthBytesUTF8(config.tokenScoresJson || '') + 1;


  const n = lmFlowLen + lmMainLen + encoderLen + decoderLen +
      textConditionerLen + vocabJsonLen + tokenScoresJsonLen;

  const buffer = Module._malloc(n);

  const len = 8 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.lmFlow || '', buffer + offset, lmFlowLen);
  offset += lmFlowLen;

  Module.stringToUTF8(config.lmMain || '', buffer + offset, lmMainLen);
  offset += lmMainLen;

  Module.stringToUTF8(config.encoder || '', buffer + offset, encoderLen);
  offset += encoderLen;

  Module.stringToUTF8(config.decoder || '', buffer + offset, decoderLen);
  offset += decoderLen;

  Module.stringToUTF8(
      config.textConditioner || '', buffer + offset, textConditionerLen);
  offset += textConditionerLen;

  Module.stringToUTF8(config.vocabJson || '', buffer + offset, vocabJsonLen);
  offset += vocabJsonLen;

  Module.stringToUTF8(
      config.tokenScoresJson || '', buffer + offset, tokenScoresJsonLen);
  offset += tokenScoresJsonLen;

  offset = 0;
  Module.setValue(ptr + 0 * 4, buffer + offset, 'i8*');
  offset += lmFlowLen;

  Module.setValue(ptr + 1 * 4, buffer + offset, 'i8*');
  offset += lmMainLen;

  Module.setValue(ptr + 2 * 4, buffer + offset, 'i8*');
  offset += encoderLen;

  Module.setValue(ptr + 3 * 4, buffer + offset, 'i8*');
  offset += decoderLen;

  Module.setValue(ptr + 4 * 4, buffer + offset, 'i8*');
  offset += textConditionerLen;

  Module.setValue(ptr + 5 * 4, buffer + offset, 'i8*');
  offset += vocabJsonLen;

  Module.setValue(ptr + 6 * 4, buffer + offset, 'i8*');
  offset += tokenScoresJsonLen;

  Module.setValue(
      ptr + 7 * 4,
      config.voiceEmbeddingCacheCapacity !== undefined ?
          config.voiceEmbeddingCacheCapacity :
          50,
      'i32');

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsSupertonicModelConfig(config, Module) {
  const durationPredictorLen =
      Module.lengthBytesUTF8(config.durationPredictor || '') + 1;
  const textEncoderLen = Module.lengthBytesUTF8(config.textEncoder || '') + 1;
  const vectorEstimatorLen =
      Module.lengthBytesUTF8(config.vectorEstimator || '') + 1;
  const vocoderLen = Module.lengthBytesUTF8(config.vocoder || '') + 1;
  const ttsJsonLen = Module.lengthBytesUTF8(config.ttsJson || '') + 1;
  const unicodeIndexerLen =
      Module.lengthBytesUTF8(config.unicodeIndexer || '') + 1;
  const voiceStyleLen = Module.lengthBytesUTF8(config.voiceStyle || '') + 1;

  const n = durationPredictorLen + textEncoderLen + vectorEstimatorLen +
      vocoderLen + ttsJsonLen + unicodeIndexerLen + voiceStyleLen;

  const buffer = Module._malloc(n);

  const len = 7 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(
      config.durationPredictor || '', buffer + offset, durationPredictorLen);
  offset += durationPredictorLen;

  Module.stringToUTF8(
      config.textEncoder || '', buffer + offset, textEncoderLen);
  offset += textEncoderLen;

  Module.stringToUTF8(
      config.vectorEstimator || '', buffer + offset, vectorEstimatorLen);
  offset += vectorEstimatorLen;

  Module.stringToUTF8(config.vocoder || '', buffer + offset, vocoderLen);
  offset += vocoderLen;

  Module.stringToUTF8(config.ttsJson || '', buffer + offset, ttsJsonLen);
  offset += ttsJsonLen;

  Module.stringToUTF8(
      config.unicodeIndexer || '', buffer + offset, unicodeIndexerLen);
  offset += unicodeIndexerLen;

  Module.stringToUTF8(config.voiceStyle || '', buffer + offset, voiceStyleLen);
  offset += voiceStyleLen;

  offset = 0;
  Module.setValue(ptr + 0 * 4, buffer + offset, 'i8*');
  offset += durationPredictorLen;

  Module.setValue(ptr + 1 * 4, buffer + offset, 'i8*');
  offset += textEncoderLen;

  Module.setValue(ptr + 2 * 4, buffer + offset, 'i8*');
  offset += vectorEstimatorLen;

  Module.setValue(ptr + 3 * 4, buffer + offset, 'i8*');
  offset += vocoderLen;

  Module.setValue(ptr + 4 * 4, buffer + offset, 'i8*');
  offset += ttsJsonLen;

  Module.setValue(ptr + 5 * 4, buffer + offset, 'i8*');
  offset += unicodeIndexerLen;

  Module.setValue(ptr + 6 * 4, buffer + offset, 'i8*');
  offset += voiceStyleLen;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsModelConfig(config, Module) {
  // Support short aliases (used by Dart web implementation).
  const aliasMap = {
    vits: 'offlineTtsVitsModelConfig',
    matcha: 'offlineTtsMatchaModelConfig',
    kokoro: 'offlineTtsKokoroModelConfig',
    kitten: 'offlineTtsKittenModelConfig',
    zipvoice: 'offlineTtsZipVoiceModelConfig',
    pocket: 'offlineTtsPocketModelConfig',
    supertonic: 'offlineTtsSupertonicModelConfig',
  };
  for (const [short, full] of Object.entries(aliasMap)) {
    if (short in config && !(full in config)) {
      config[full] = config[short];
    }
  }

  if (!('offlineTtsVitsModelConfig' in config)) {
    config.offlineTtsVitsModelConfig = {
      model: '',
      lexicon: '',
      tokens: '',
      noiseScale: 0.667,
      noiseScaleW: 0.8,
      lengthScale: 1.0,
      dataDir: '',
    };
  }

  if (!('offlineTtsMatchaModelConfig' in config)) {
    config.offlineTtsMatchaModelConfig = {
      acousticModel: '',
      vocoder: '',
      lexicon: '',
      tokens: '',
      noiseScale: 0.667,
      lengthScale: 1.0,
      dataDir: '',
    };
  }

  if (!('offlineTtsKokoroModelConfig' in config)) {
    config.offlineTtsKokoroModelConfig = {
      model: '',
      voices: '',
      tokens: '',
      lengthScale: 1.0,
      dataDir: '',
      lexicon: '',
      lang: '',
    };
  }

  if (!('offlineTtsKittenModelConfig' in config)) {
    config.offlineTtsKittenModelConfig = {
      model: '',
      voices: '',
      tokens: '',
      lengthScale: 1.0,
    };
  }

  if (!('offlineTtsZipVoiceModelConfig' in config)) {
    config.offlineTtsZipVoiceModelConfig = {
      tokens: '',
      encoder: '',
      decoder: '',
      vocoder: '',
      dataDir: '',
      lexicon: '',
      featScale: 0.1,
      tShift: 0.5,
      targetRMS: 0.1,
      guidanceScale: 1.0,
    };
  }

  if (!('offlineTtsPocketModelConfig' in config)) {
    config.offlineTtsPocketModelConfig = {
      lmFlow: '',
      lmMain: '',
      encoder: '',
      decoder: '',
      textConditioner: '',
      vocabJson: '',
      tokenScoresJson: '',
      voiceEmbeddingCacheCapacity: 50,
    };
  }

  if (!('offlineTtsSupertonicModelConfig' in config)) {
    config.offlineTtsSupertonicModelConfig = {
      durationPredictor: '',
      textEncoder: '',
      vectorEstimator: '',
      vocoder: '',
      ttsJson: '',
      unicodeIndexer: '',
      voiceStyle: '',
    };
  }

  const vitsModelConfig = initSherpaOnnxOfflineTtsVitsModelConfig(
      config.offlineTtsVitsModelConfig, Module);

  const matchaModelConfig = initSherpaOnnxOfflineTtsMatchaModelConfig(
      config.offlineTtsMatchaModelConfig, Module);

  const kokoroModelConfig = initSherpaOnnxOfflineTtsKokoroModelConfig(
      config.offlineTtsKokoroModelConfig, Module);

  const kittenModelConfig = initSherpaOnnxOfflineTtsKittenModelConfig(
      config.offlineTtsKittenModelConfig, Module);

  const zipVoiceModelConfig = initSherpaOnnxOfflineTtsZipVoiceModelConfig(
      config.offlineTtsZipVoiceModelConfig, Module);

  const pocketModelConfig = initSherpaOnnxOfflineTtsPocketModelConfig(
      config.offlineTtsPocketModelConfig, Module);

  const supertonicModelConfig = initSherpaOnnxOfflineTtsSupertonicModelConfig(
      config.offlineTtsSupertonicModelConfig, Module);

  const len = vitsModelConfig.len + matchaModelConfig.len +
      kokoroModelConfig.len + kittenModelConfig.len + zipVoiceModelConfig.len +
      pocketModelConfig.len + supertonicModelConfig.len + 3 * 4;

  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(vitsModelConfig.ptr, vitsModelConfig.len, ptr + offset);
  offset += vitsModelConfig.len;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.debug || 0, 'i32');
  offset += 4;

  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const buffer = Module._malloc(providerLen);
  Module.stringToUTF8(config.provider || 'cpu', buffer, providerLen);
  Module.setValue(ptr + offset, buffer, 'i8*');
  offset += 4;

  Module._CopyHeap(matchaModelConfig.ptr, matchaModelConfig.len, ptr + offset);
  offset += matchaModelConfig.len;

  Module._CopyHeap(kokoroModelConfig.ptr, kokoroModelConfig.len, ptr + offset);
  offset += kokoroModelConfig.len;

  Module._CopyHeap(kittenModelConfig.ptr, kittenModelConfig.len, ptr + offset);
  offset += kittenModelConfig.len;

  Module._CopyHeap(
      zipVoiceModelConfig.ptr, zipVoiceModelConfig.len, ptr + offset);
  offset += zipVoiceModelConfig.len;

  Module._CopyHeap(pocketModelConfig.ptr, pocketModelConfig.len, ptr + offset);
  offset += pocketModelConfig.len;

  Module._CopyHeap(
      supertonicModelConfig.ptr, supertonicModelConfig.len, ptr + offset);
  offset += supertonicModelConfig.len;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
    config: vitsModelConfig,
    matcha: matchaModelConfig,
    kokoro: kokoroModelConfig,
    kitten: kittenModelConfig,
    zipvoice: zipVoiceModelConfig,
    pocket: pocketModelConfig,
    supertonic: supertonicModelConfig,
  };
}

function initSherpaOnnxOfflineTtsConfig(config, Module) {
  // Support short alias for model config (used by Dart web implementation).
  if ('model' in config && !('offlineTtsModelConfig' in config)) {
    config.offlineTtsModelConfig = config.model;
  }

  const modelConfig =
      initSherpaOnnxOfflineTtsModelConfig(config.offlineTtsModelConfig, Module);
  const len = modelConfig.len + 4 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset);
  offset += modelConfig.len;

  const ruleFstsLen = Module.lengthBytesUTF8(config.ruleFsts || '') + 1;
  const ruleFarsLen = Module.lengthBytesUTF8(config.ruleFars || '') + 1;

  const buffer = Module._malloc(ruleFstsLen + ruleFarsLen);
  Module.stringToUTF8(config.ruleFsts || '', buffer, ruleFstsLen);
  Module.stringToUTF8(config.ruleFars || '', buffer + ruleFstsLen, ruleFarsLen);

  Module.setValue(ptr + offset, buffer, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.maxNumSentences || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + ruleFstsLen, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.silenceScale || 0.2, 'float');
  offset += 4;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
    config: modelConfig,
  };
}

/*
const genConfig = {
  silenceScale: 0.2,
  speed: 1.0,
  sid: 1,
  referenceAudio: myFloat32Array, // optional
  referenceSampleRate: 16000, // used if referenceAudio is required
  referenceText: "Hello world", // optional
  numSteps: 5, // optional
  extra: { bar: "ok", foo: 0.8, foobar: 10}
};

 */

// Validate a nested-array field (phonemeCodepoints: number[][], tokens:
// string[][]) before anything is allocated. Anything else is a caller bug,
// so throw a TypeError naming the offending element.
function validateNestedArray(name, value, elementType) {
  if (!Array.isArray(value)) {
    throw new TypeError(`${name} must be an array of arrays`);
  }
  for (let i = 0; i < value.length; ++i) {
    const inner = value[i];
    if (!Array.isArray(inner)) {
      throw new TypeError(`${name}[${i}] must be an array`);
    }
    if (inner.length === 0) {
      throw new TypeError(`${name}[${i}] must be a non-empty array`);
    }
    for (let j = 0; j < inner.length; ++j) {
      const item = inner[j];
      const ok = elementType === 'number' ? typeof item === 'number' :
          typeof item === 'string';
      if (!ok) {
        throw new TypeError(`${name}[${i}][${j}] must be a ${elementType}`);
      }
    }
  }
}


// Allocate a SherpaOnnxGenerationConfig in WASM
function initSherpaOnnxGenerationConfig(config, Module) {
  // Validate before allocating anything.
  if (config.phonemeCodepoints !== undefined) {
    validateNestedArray('phonemeCodepoints', config.phonemeCodepoints, 'number');
  }
  if (config.tokens !== undefined) {
    validateNestedArray('tokens', config.tokens, 'string');
  }

  const len = 15 * 4;
  const ptr = Module._malloc(len);

  // float silence_scale
  Module.setValue(ptr + 0 * 4, config.silenceScale || 0.2, 'float');

  // float speed
  Module.setValue(ptr + 1 * 4, config.speed || 1.0, 'float');

  // int32_t sid
  Module.setValue(ptr + 2 * 4, config.sid || 0, 'i32');

  // const float* reference_audio
  let referenceAudioPtr = 0;
  if (config.referenceAudio && config.referenceAudio.length > 0) {
    referenceAudioPtr = Module._malloc(config.referenceAudio.length * 4);
    Module.HEAPF32.set(config.referenceAudio, referenceAudioPtr / 4);
  }
  Module.setValue(ptr + 3 * 4, referenceAudioPtr, 'i8*');

  // int32_t reference_audio_len
  Module.setValue(
      ptr + 4 * 4, config.referenceAudio ? config.referenceAudio.length : 0,
      'i32');

  // int32_t reference_sample_rate
  Module.setValue(ptr + 5 * 4, config.referenceSampleRate || 0, 'i32');

  // const char* reference_text
  let referenceTextPtr = 0;
  if (config.referenceText) {
    const textLen = Module.lengthBytesUTF8(config.referenceText) + 1;
    referenceTextPtr = Module._malloc(textLen);
    Module.stringToUTF8(config.referenceText, referenceTextPtr, textLen);
  }
  Module.setValue(ptr + 6 * 4, referenceTextPtr, 'i8*');

  // int32_t num_steps
  Module.setValue(ptr + 7 * 4, config.numSteps || 5, 'i32');

  // const char* extra (JSON string)
  let extraPtr = 0;
  let extraStr = null;

  if (config.extra) {
    if (typeof config.extra === 'object') {
      extraStr = JSON.stringify(config.extra);
    } else if (typeof config.extra === 'string') {
      extraStr = config.extra;
    }
  }

  if (extraStr !== null) {
    const extraLen = Module.lengthBytesUTF8(extraStr) + 1;
    extraPtr = Module._malloc(extraLen);
    Module.stringToUTF8(extraStr, extraPtr, extraLen);
  }

  Module.setValue(ptr + 8 * 4, extraPtr, 'i8*');

  // const int32_t* phoneme_codepoints + lens + num_sentences
  let phonemeCodepointsPtr = 0;
  let phonemeCodepointsLensPtr = 0;
  if (config.phonemeCodepoints && config.phonemeCodepoints.length > 0) {
    const numSentences = config.phonemeCodepoints.length;
    let total = 0;
    for (const inner of config.phonemeCodepoints) total += inner.length;

    phonemeCodepointsPtr = Module._malloc(total * 4);
    phonemeCodepointsLensPtr = Module._malloc(numSentences * 4);

    const flat = new Int32Array(total);
    const lens = new Int32Array(numSentences);
    let offset = 0;
    for (let i = 0; i < numSentences; ++i) {
      flat.set(config.phonemeCodepoints[i], offset);
      lens[i] = config.phonemeCodepoints[i].length;
      offset += config.phonemeCodepoints[i].length;
    }
    Module.HEAP32.set(flat, phonemeCodepointsPtr / 4);
    Module.HEAP32.set(lens, phonemeCodepointsLensPtr / 4);
  }
  Module.setValue(ptr + 9 * 4, phonemeCodepointsPtr, 'i8*');
  Module.setValue(ptr + 10 * 4, phonemeCodepointsLensPtr, 'i8*');
  Module.setValue(ptr + 11 * 4,
      config.phonemeCodepoints ? config.phonemeCodepoints.length : 0, 'i32');

  // const char* const* tokens + lens + num_sentences
  let tokensPtr = 0;
  let tokensLensPtr = 0;
  const tokenStringPtrs = [];
  if (config.tokens && config.tokens.length > 0) {
    const numSentences = config.tokens.length;
    let total = 0;
    for (const inner of config.tokens) total += inner.length;

    tokensPtr = Module._malloc(Math.max(total, 1) * 4);
    tokensLensPtr = Module._malloc(numSentences * 4);

    const lens = new Int32Array(numSentences);
    let offset = 0;
    for (let i = 0; i < numSentences; ++i) {
      lens[i] = config.tokens[i].length;
      for (const token of config.tokens[i]) {
        const n = Module.lengthBytesUTF8(token) + 1;
        const p = Module._malloc(n);
        Module.stringToUTF8(token, p, n);
        tokenStringPtrs.push(p);
        Module.setValue(tokensPtr + offset * 4, p, 'i8*');
        offset += 1;
      }
    }
    Module.HEAP32.set(lens, tokensLensPtr / 4);
  }
  Module.setValue(ptr + 12 * 4, tokensPtr, 'i8*');
  Module.setValue(ptr + 13 * 4, tokensLensPtr, 'i8*');
  Module.setValue(ptr + 14 * 4, config.tokens ? config.tokens.length : 0,
      'i32');

  return {
    ptr,
    referenceAudioPtr,
    referenceTextPtr,
    extraPtr,
    phonemeCodepointsPtr,
    phonemeCodepointsLensPtr,
    tokensPtr,
    tokensLensPtr,
    tokenStringPtrs,
  };
}


// Free the memory allocated for a SherpaOnnxGenerationConfig
function freeSherpaOnnxGenerationConfig(cfg, Module) {
  if (!cfg) return;

  if (cfg.referenceAudioPtr) Module._free(cfg.referenceAudioPtr);
  if (cfg.referenceTextPtr) Module._free(cfg.referenceTextPtr);
  if (cfg.extraPtr) Module._free(cfg.extraPtr);
  if (cfg.phonemeCodepointsPtr) Module._free(cfg.phonemeCodepointsPtr);
  if (cfg.phonemeCodepointsLensPtr) Module._free(cfg.phonemeCodepointsLensPtr);
  if (cfg.tokensPtr) Module._free(cfg.tokensPtr);
  if (cfg.tokensLensPtr) Module._free(cfg.tokensLensPtr);
  for (const p of (cfg.tokenStringPtrs || [])) Module._free(p);
  if (cfg.ptr) Module._free(cfg.ptr);
}


class OfflineTts {
  constructor(configObj, Module) {
    const config = initSherpaOnnxOfflineTtsConfig(configObj, Module)
    const handle = Module._SherpaOnnxCreateOfflineTts(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.sampleRate = Module._SherpaOnnxOfflineTtsSampleRate(this.handle);
    this.numSpeakers = Module._SherpaOnnxOfflineTtsNumSpeakers(this.handle);
    {
      const p = Module._SherpaOnnxOfflineTtsLang(this.handle);
      this.lang = p ? Module.UTF8ToString(p) : '';
      Module._SherpaOnnxDestroyOfflineTtsLang(p);
    }
    this.Module = Module
  }

  free() {
    if (!this.handle) return;

    this.Module._SherpaOnnxDestroyOfflineTts(this.handle);
    this.handle = 0
  }

  // {
  //   text: "hello",
  //   sid: 1,
  //   speed: 1.0
  // }
  generate(config) {
    if (!this.handle) {
      throw new Error('OfflineTts has been freed');
    }

    if (!config || !config.text) {
      throw new Error('config.text is required');
    }

    const textLen = this.Module.lengthBytesUTF8(config.text) + 1;
    const textPtr = this.Module._malloc(textLen);
    this.Module.stringToUTF8(config.text, textPtr, textLen);

    const genConfig = {
      sid: config.sid ?? 0,
      speed: config.speed ?? 1.0,
    };
    const cfgWasm = initSherpaOnnxGenerationConfig(genConfig, this.Module);

    const h = this.Module._SherpaOnnxOfflineTtsGenerateWithConfig(
        this.handle, textPtr, cfgWasm.ptr, 0, 0);

    freeSherpaOnnxGenerationConfig(cfgWasm, this.Module);
    this.Module._free(textPtr);

    if (!h) {
      throw new Error('TTS generation failed');
    }

    const base = h / 4;

    const samplesPtr = this.Module.HEAPU32[base];
    const numSamples = this.Module.HEAP32[base + 1];
    const sampleRate = this.Module.HEAP32[base + 2];

    const heapSamples = this.Module.HEAPF32.subarray(
        samplesPtr / 4, samplesPtr / 4 + numSamples);

    const samples = new Float32Array(heapSamples);

    this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(h);
    return {samples: samples, sampleRate: sampleRate};
  }

  generateWithConfig(text, genConfig) {
    if (!this.handle) {
      throw new Error('OfflineTts has been freed');
    }

    const cfgWasm = initSherpaOnnxGenerationConfig(genConfig, this.Module);

    const textLen = this.Module.lengthBytesUTF8(text) + 1;
    const textPtr = this.Module._malloc(textLen);
    this.Module.stringToUTF8(text, textPtr, textLen);

    let callbackPtr = 0;
    if (genConfig.callback) {
      callbackPtr = this.Module.addFunction((samplesPtr, n, progress, arg) => {
        const heapSamples =
            this.Module.HEAPF32.subarray(samplesPtr / 4, samplesPtr / 4 + n);
        const samples = new Float32Array(heapSamples);
        return genConfig.callback(samples, n, progress, arg);
      }, 'iiifi');
    }

    let audioPtr = 0;
    try {
      audioPtr = this.Module._SherpaOnnxOfflineTtsGenerateWithConfig(
          this.handle, textPtr, cfgWasm.ptr, callbackPtr, 0);
    } finally {
      this.Module._free(textPtr);
      freeSherpaOnnxGenerationConfig(cfgWasm, this.Module);
      if (callbackPtr) {
        this.Module.removeFunction(callbackPtr);
      }
    }

    if (!audioPtr) {
      // Generation failed. The C API already logged the reason (for instance
      // phonemeCodepoints and tokens are mutually exclusive). Return empty
      // audio instead of throwing, matching C++/C/Python/Rust/Go/Java.
      return {samples: new Float32Array(0), sampleRate: 0};
    }

    const base = audioPtr / 4;

    const samplesPtr = this.Module.HEAPU32[base];     // float* samples
    const numSamples = this.Module.HEAP32[base + 1];  // int32 num_samples
    const sampleRate = this.Module.HEAP32[base + 2];  // int32 sample_rate

    const heapSamples = this.Module.HEAPF32.subarray(
        samplesPtr / 4, samplesPtr / 4 + numSamples);
    const samples = new Float32Array(heapSamples);

    this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(audioPtr);

    return {samples, sampleRate};
  }

  save(filename, audio) {
    const samples = audio.samples;
    const sampleRate = audio.sampleRate;
    const ptr = this.Module._malloc(samples.length * 4);

    this.Module.HEAPF32.set(samples, ptr / 4);

    const filenameLen = this.Module.lengthBytesUTF8(filename) + 1;
    const buffer = this.Module._malloc(filenameLen);
    this.Module.stringToUTF8(filename, buffer, filenameLen);
    this.Module._SherpaOnnxWriteWave(ptr, samples.length, sampleRate, buffer);
    this.Module._free(buffer);
    this.Module._free(ptr);
  }
}

let modelType = 0;

// Choose exactly one of the two text frontends below (not both):
//
// usePhonemize = true // option 1 (default): phonemize the text with piper-phonemize; no extra files needed
// lexicon = './lexicon-en-us.txt' // option 2: lexicon file next to the model files.
//   Download it from https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
//
// Neither option uses the espeak-ng-data directory shipped in the model
// package; you can leave it out to keep the download small.
//
// Option 1 loads the piper-phonemize wasm bundle from ./piper-phonemize/
// (see assets/README.md for how to get it).
//
// The demo UI can flip this at runtime (the TTS instance is recreated with
// the chosen frontend).
let usePhonemize = true;

function setUsePhonemize(value) {
  usePhonemize = value;
}

// The frontend the selected model actually uses. Models that bring their own
// lexicon (Chinese/multi-lingual/zipvoice) always use it; the German model is
// phonemize-only because there is no German lexicon.
let modelUsesPhonemize = false;

function getDefaultOfflineTtsModelType() {
  return modelType;
}

function getDefaultUsePhonemize() {
  return modelUsesPhonemize;
}

function createOfflineTts(Module, myConfig) {
  const vits = {
    model: '',
    lexicon: '',
    tokens: '',
    dataDir: '',  // deprecated, ignored; use lexicon or phonemeCodepoints
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  };

  const matcha = {
    acousticModel: '',
    vocoder: '',
    lexicon: '',
    tokens: '',
    dataDir: '',  // deprecated, ignored; use lexicon or phonemeCodepoints
    noiseScale: 0.667,
    lengthScale: 1.0,
  };

  const offlineTtsKokoroModelConfig = {
    model: '',
    voices: '',
    tokens: '',
    dataDir: '',  // deprecated, ignored; use lexicon or phonemeCodepoints
    lengthScale: 1.0,
    lexicon: '',
    lang: '',
  };

  const offlineTtsKittenModelConfig = {
    model: '',
    voices: '',
    tokens: '',
    dataDir: '',  // deprecated, ignored; use lexicon or phonemeCodepoints
    lengthScale: 1.0,
    lexicon: '',
  };

  const offlineTtsZipVoiceModelConfig = {
    tokens: '',
    encoder: '',
    decoder: '',
    vocoder: '',
    dataDir: '',  // deprecated, ignored; use lexicon or phonemeCodepoints
    lexicon: '',
    featScale: 0.1,
    tShift: 0.5,
    targetRMS: 0.1,
    guidanceScale: 1.0,
  };

  const offlineTtsPocketModelConfig = {
    lmFlow: '',
    lmMain: '',
    encoder: '',
    decoder: '',
    textConditioner: '',
    vocabJson: '',
    tokenScoresJson: '',
    voiceEmbeddingCacheCapacity: 50,
  };

  const offlineTtsSupertonicModelConfig = {
    durationPredictor: '',
    textEncoder: '',
    vectorEstimator: '',
    vocoder: '',
    ttsJson: '',
    unicodeIndexer: '',
    voiceStyle: '',
  };

  let ruleFsts = '';

  switch (modelType) {
    case 0:
      // vits
      vits.model = './model.onnx';
      vits.tokens = './tokens.txt';
      // Record the frontend actually configured: the ready message reports
      // this back and the UI keeps the picker in sync with it.
      modelUsesPhonemize = usePhonemize;
      if (!usePhonemize) {
        vits.lexicon = './lexicon-en-us.txt';
      }
      break;
    case 1:
      // matcha zh-en
      // https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/matcha-icefall-zh-en.html
      matcha.acousticModel = './model-steps-3.onnx';
      matcha.vocoder = './vocos-16khz-univ.onnx';
      matcha.lexicon = './lexicon.txt';
      matcha.tokens = './tokens.txt';
      ruleFsts = './phone-zh.fst,./date-zh.fst,./number-zh.fst';
      break;
    case 2:
      // matcha zh
      // https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese/matcha-icefall-zh-baker.html
      matcha.acousticModel = './model-steps-3.onnx';
      matcha.vocoder = './vocos-22khz-univ.onnx';
      matcha.lexicon = './lexicon.txt';
      matcha.tokens = './tokens.txt';
      ruleFsts = './phone.fst,./date.fst,./number.fst';
      break;
    case 3:
      // matcha en
      // https://k2-fsa.github.io/sherpa/onnx/tts/all/English/matcha-icefall-en_US-ljspeech.html
      matcha.acousticModel = './model-steps-3.onnx';
      matcha.vocoder = './vocos-22khz-univ.onnx';
      matcha.tokens = './tokens.txt';
      modelUsesPhonemize = usePhonemize;
      if (!usePhonemize) {
        matcha.lexicon = './lexicon-en-us.txt';
      }
      break;
    case 4:
      // zipvoice zh-en
      // https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html
      offlineTtsZipVoiceModelConfig.tokens = './tokens.txt';
      offlineTtsZipVoiceModelConfig.encoder = './encoder.int8.onnx';
      offlineTtsZipVoiceModelConfig.decoder = './decoder.int8.onnx';
      offlineTtsZipVoiceModelConfig.vocoder = './vocos_24khz.onnx';
      offlineTtsZipVoiceModelConfig.lexicon = './lexicon.txt';
      break;
    case 5:
      // pocket tts
      // https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
      offlineTtsPocketModelConfig.lmFlow = './lm_flow.int8.onnx';
      offlineTtsPocketModelConfig.lmMain = './lm_main.int8.onnx';
      offlineTtsPocketModelConfig.encoder = './encoder.onnx';
      offlineTtsPocketModelConfig.decoder = './decoder.int8.onnx';
      offlineTtsPocketModelConfig.textConditioner = './text_conditioner.onnx';
      offlineTtsPocketModelConfig.vocabJson = './vocab.json';
      offlineTtsPocketModelConfig.tokenScoresJson = './token_scores.json';
      break;
    case 6:
      // KittenTTS
      offlineTtsKittenModelConfig.model = './model.onnx';
      offlineTtsKittenModelConfig.voices = './voices.bin';
      offlineTtsKittenModelConfig.tokens = './tokens.txt';
      modelUsesPhonemize = usePhonemize;
      if (!usePhonemize) {
        offlineTtsKittenModelConfig.lexicon = './lexicon-en-us.txt';
      }
      break;
    case 7:
      // SupertonicTTS (multilingual)
      // https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
      offlineTtsSupertonicModelConfig.durationPredictor =
          './duration_predictor.int8.onnx';
      offlineTtsSupertonicModelConfig.textEncoder = './text_encoder.int8.onnx';
      offlineTtsSupertonicModelConfig.vectorEstimator =
          './vector_estimator.int8.onnx';
      offlineTtsSupertonicModelConfig.vocoder = './vocoder.int8.onnx';
      offlineTtsSupertonicModelConfig.ttsJson = './tts.json';
      offlineTtsSupertonicModelConfig.unicodeIndexer =
          './unicode_indexer.bin';
      offlineTtsSupertonicModelConfig.voiceStyle = './voice.bin';
      // Own text frontend (unicode indexer); not the phonemize/lexicon picker.
      modelUsesPhonemize = false;
      break;
    case 8:
      // kokoro-multi-lang-v1_0
      // https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/kokoro-multi-lang-v1_0.html
      // Note: case 8, not 7. 7 is Supertonic's id and must stay unique.
      offlineTtsKokoroModelConfig.model = './model.onnx';
      offlineTtsKokoroModelConfig.voices = './voices.bin';
      offlineTtsKokoroModelConfig.tokens = './tokens.txt';
      // data_dir is ignored since v2.0.0; the lexicon below is the frontend.
      offlineTtsKokoroModelConfig.lexicon = './lexicon-us-en.txt,./lexicon-zh.txt';
      ruleFsts = './phone-zh.fst,./date-zh.fst,./number-zh.fst';
      break;
  }

  const offlineTtsModelConfig = {
    offlineTtsVitsModelConfig: vits,
    offlineTtsMatchaModelConfig: matcha,
    offlineTtsKokoroModelConfig: offlineTtsKokoroModelConfig,
    offlineTtsKittenModelConfig: offlineTtsKittenModelConfig,
    offlineTtsZipVoiceModelConfig: offlineTtsZipVoiceModelConfig,
    offlineTtsPocketModelConfig: offlineTtsPocketModelConfig,
    offlineTtsSupertonicModelConfig: offlineTtsSupertonicModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };

  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    ruleFsts: ruleFsts,
    ruleFars: '',
    maxNumSentences: 1,
  }

  if (myConfig) {
    offlineTtsConfig = myConfig;
  }

  return new OfflineTts(offlineTtsConfig, Module);
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createOfflineTts,
    getDefaultOfflineTtsModelType,
  };
}
