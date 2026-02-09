
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

  Module._free(config.ptr);
}

// The user should free the returned pointers
function initSherpaOnnxOfflineTtsVitsModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
  const dictDir = ''
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

  const n = modelLen + voicesLen + tokensLen + dataDirLen;

  const buffer = Module._malloc(n);

  const len = 5 * 4;
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

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsZipVoiceModelConfig(config, Module) {
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

  const len = 7 * 4;
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

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineTtsModelConfig(config, Module) {
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

  const len = vitsModelConfig.len + matchaModelConfig.len +
      kokoroModelConfig.len + kittenModelConfig.len + zipVoiceModelConfig.len +
      pocketModelConfig.len + 3 * 4;

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
  };
}

function initSherpaOnnxOfflineTtsConfig(config, Module) {
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
  referenceSample_rate: 16000, // used if referenceAudio is required
  referenceText: "Hello world", // optional
  numSteps: 5, // optional
  extra: { bar: "ok", foo: 0.8, foobar: 10}
};
};

 */

// Allocate a SherpaOnnxGenerationConfig in WASM
function initSherpaOnnxGenerationConfig(config, Module) {
  const len = 9 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;

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

  // const char* extra
  let extraPtr = 0;

  if (config.extra && typeof config.extra === 'object') {
    config.extra = JSON.stringify(config.extra);

    const extraLen = Module.lengthBytesUTF8(config.extra) + 1;
    extraPtr = Module._malloc(extraLen);
    Module.stringToUTF8(config.extra, extraPtr, extraLen);
  }

  Module.setValue(ptr + 8 * 4, extraPtr, 'i8*');

  return {
    ptr,
    referenceAudioPtr,
    referenceTextPtr,
    extraPtr,
  };
}

// Free the memory allocated for a SherpaOnnxGenerationConfig
function freeSherpaOnnxGenerationConfig(cfg, Module) {
  if (!cfg) return;

  if (cfg.referenceAudioPtr) Module._free(cfg.referenceAudioPtr);
  if (cfg.referenceTextPtr) Module._free(cfg.referenceTextPtr);
  if (cfg.extraPtr) Module._free(cfg.extraPtr);
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
    this.Module = Module
  }

  free() {
    this.Module._SherpaOnnxDestroyOfflineTts(this.handle);
    this.handle = 0
  }

  // {
  //   text: "hello",
  //   sid: 1,
  //   speed: 1.0
  // }
  generate(config) {
    const textLen = this.Module.lengthBytesUTF8(config.text) + 1;
    const textPtr = this.Module._malloc(textLen);
    this.Module.stringToUTF8(config.text, textPtr, textLen);

    const h = this.Module._SherpaOnnxOfflineTtsGenerate(
        this.handle, textPtr, config.sid, config.speed);

    const numSamples = this.Module.HEAP32[h / 4 + 1];
    const sampleRate = this.Module.HEAP32[h / 4 + 2];

    const samplesPtr = this.Module.HEAP32[h / 4] / 4;
    const samples = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      samples[i] = this.Module.HEAPF32[samplesPtr + i];
    }

    this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(h);
    return {samples: samples, sampleRate: sampleRate};
  }

  generateWithConfig(text, genConfig) {
    // 1️⃣ Allocate SherpaOnnxGenerationConfig in WASM
    const cfgWasm = initSherpaOnnxGenerationConfig(genConfig, this.Module);

    // 2️⃣ Allocate text in WASM
    const textLen = this.Module.lengthBytesUTF8(text) + 1;
    const textPtr = this.Module._malloc(textLen);
    this.Module.stringToUTF8(text, textPtr, textLen);

    // 3️⃣ Call the C API
    const audioPtr = this.Module._SherpaOnnxOfflineTtsGenerateWithConfig(
        this.handle, textPtr, cfgWasm.ptr,
        0,  // callback
        0   // callback arg
    );

    if (!audioPtr) {
      this.Module._free(textPtr);
      freeSherpaOnnxGenerationConfig(cfgWasm, this.Module);
      throw new Error('Failed to generate audio');
    }

    // 4️⃣ Read SherpaOnnxGeneratedAudio struct
    const samplesPtr = this.Module.HEAP32[audioPtr / 4];  // float* samples
    const numSamples =
        this.Module.HEAP32[audioPtr / 4 + 1];  // int32 num_samples
    const sampleRate =
        this.Module.HEAP32[audioPtr / 4 + 2];  // int32 sample_rate

    // 5️⃣ Copy samples to Float32Array
    const samples = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      samples[i] = this.Module.HEAPF32[samplesPtr / 4 + i];
    }

    // 6️⃣ Free WASM memory
    this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(audioPtr);
    this.Module._free(textPtr);
    freeSherpaOnnxGenerationConfig(cfgWasm, this.Module);

    return {samples, sampleRate};
  }

  save(filename, audio) {
    const samples = audio.samples;
    const sampleRate = audio.sampleRate;
    const ptr = this.Module._malloc(samples.length * 4);
    for (let i = 0; i < samples.length; i++) {
      this.Module.HEAPF32[ptr / 4 + i] = samples[i];
    }

    const filenameLen = this.Module.lengthBytesUTF8(filename) + 1;
    const buffer = this.Module._malloc(filenameLen);
    this.Module.stringToUTF8(filename, buffer, filenameLen);
    this.Module._SherpaOnnxWriteWave(ptr, samples.length, sampleRate, buffer);
    this.Module._free(buffer);
    this.Module._free(ptr);
  }
}

function createOfflineTts(Module, myConfig) {
  const vits = {
    model: '',
    lexicon: '',
    tokens: '',
    dataDir: '',
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  };

  const matcha = {
    acousticModel: '',
    vocoder: '',
    lexicon: '',
    tokens: '',
    dataDir: '',
    noiseScale: 0.667,
    lengthScale: 1.0,
  };

  const offlineTtsKokoroModelConfig = {
    model: '',
    voices: '',
    tokens: '',
    dataDir: '',
    lengthScale: 1.0,
    lexicon: '',
    lang: '',
  };

  const offlineTtsKittenModelConfig = {
    model: '',
    voices: '',
    tokens: '',
    dataDir: '',
    lengthScale: 1.0,
  };

  let ruleFsts = '';

  let type = 0;
  switch (type) {
    case 0:
      // vits
      vits.model = './model.onnx';
      vits.tokens = './tokens.txt';
      vits.dataDir = './espeak-ng-data';
      break;
    case 1:
      // matcha zh-en
      // https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/matcha-icefall-zh-en.html
      matcha.acousticModel = './model-steps-3.onnx';
      matcha.vocoder = './vocos-16khz-univ.onnx';
      matcha.lexicon = './lexicon.txt';
      matcha.tokens = './tokens.txt';
      matcha.dataDir = './espeak-ng-data';
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
      matcha.dataDir = './espeak-ng-data';
      break;
  }

  const offlineTtsModelConfig = {
    offlineTtsVitsModelConfig: vits,
    offlineTtsMatchaModelConfig: matcha,
    offlineTtsKokoroModelConfig: offlineTtsKokoroModelConfig,
    offlineTtsKittenModelConfig: offlineTtsKittenModelConfig,
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
  };
}
