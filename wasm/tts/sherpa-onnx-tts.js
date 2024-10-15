
function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('config' in config) {
    freeConfig(config.config, Module)
  }

  Module._free(config.ptr);
}

// The user should free the returned pointers
function initSherpaOnnxOfflineTtsVitsModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model) + 1;
  const lexiconLen = Module.lengthBytesUTF8(config.lexicon || '') + 1;
  const tokensLen = Module.lengthBytesUTF8(config.tokens || '') + 1;
  const dataDirLen = Module.lengthBytesUTF8(config.dataDir || '') + 1;
  const dictDirLen = Module.lengthBytesUTF8(config.dictDir || '') + 1;

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

  Module.stringToUTF8(config.dictDir || '', buffer + offset, dictDirLen);
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
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineTtsModelConfig(config, Module) {
  const vitsModelConfig = initSherpaOnnxOfflineTtsVitsModelConfig(
      config.offlineTtsVitsModelConfig, Module);

  const len = vitsModelConfig.len + 3 * 4;
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
  Module.stringToUTF8(config.provider, buffer, providerLen);
  Module.setValue(ptr + offset, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len, config: vitsModelConfig,
  }
}

function initSherpaOnnxOfflineTtsConfig(config, Module) {
  const modelConfig =
      initSherpaOnnxOfflineTtsModelConfig(config.offlineTtsModelConfig, Module);
  const len = modelConfig.len + 3 * 4;
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

  return {
    buffer: buffer, ptr: ptr, len: len, config: modelConfig,
  }
}

class OfflineTts {
  constructor(configObj, Module) {
    console.log(configObj)
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
  const offlineTtsVitsModelConfig = {
    model: './model.onnx',
    lexicon: '',
    tokens: './tokens.txt',
    dataDir: './espeak-ng-data',
    dictDir: '',
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  };
  const offlineTtsModelConfig = {
    offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };
  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    ruleFsts: '',
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
