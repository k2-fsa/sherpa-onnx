
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
  let modelLen = Module.lengthBytesUTF8(config.model) + 1;
  let lexiconLen = Module.lengthBytesUTF8(config.lexicon) + 1;
  let tokensLen = Module.lengthBytesUTF8(config.tokens) + 1;
  let dataDirLen = Module.lengthBytesUTF8(config.dataDir) + 1;

  let n = modelLen + lexiconLen + tokensLen + dataDirLen;

  let buffer = Module._malloc(n);

  let len = 7 * 4;
  let ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model, buffer + offset, modelLen);
  offset += modelLen;

  Module.stringToUTF8(config.lexicon, buffer + offset, lexiconLen);
  offset += lexiconLen;

  Module.stringToUTF8(config.tokens, buffer + offset, tokensLen);
  offset += tokensLen;

  Module.stringToUTF8(config.dataDir, buffer + offset, dataDirLen);
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

function initSherpaOnnxOfflineTtsModelConfig(config, Module) {
  let vitsModelConfig = initSherpaOnnxOfflineTtsVitsModelConfig(
      config.offlineTtsVitsModelConfig, Module);

  let len = vitsModelConfig.len + 3 * 4;
  let ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(vitsModelConfig.ptr, vitsModelConfig.len, ptr + offset);
  offset += vitsModelConfig.len;

  Module.setValue(ptr + offset, config.numThreads, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.debug, 'i32');
  offset += 4;

  let providerLen = Module.lengthBytesUTF8(config.provider) + 1;
  let buffer = Module._malloc(providerLen);
  Module.stringToUTF8(config.provider, buffer, providerLen);
  Module.setValue(ptr + offset, buffer, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len, config: vitsModelConfig,
  }
}

function initSherpaOnnxOfflineTtsConfig(config, Module) {
  let modelConfig =
      initSherpaOnnxOfflineTtsModelConfig(config.offlineTtsModelConfig, Module);
  let len = modelConfig.len + 2 * 4;
  let ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset);
  offset += modelConfig.len;

  let ruleFstsLen = Module.lengthBytesUTF8(config.ruleFsts) + 1;
  let buffer = Module._malloc(ruleFstsLen);
  Module.stringToUTF8(config.ruleFsts, buffer, ruleFstsLen);
  Module.setValue(ptr + offset, buffer, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.maxNumSentences, 'i32');

  return {
    buffer: buffer, ptr: ptr, len: len, config: modelConfig,
  }
}

class OfflineTts {
  constructor(configObj, Module) {
    console.log(configObj)
    let config = initSherpaOnnxOfflineTtsConfig(configObj, Module)
    let handle = Module._SherpaOnnxCreateOfflineTts(config.ptr);

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
    let textLen = this.Module.lengthBytesUTF8(config.text) + 1;
    let textPtr = this.Module._malloc(textLen);
    this.Module.stringToUTF8(config.text, textPtr, textLen);

    let h = this.Module._SherpaOnnxOfflineTtsGenerate(
        this.handle, textPtr, config.sid, config.speed);

    let numSamples = this.Module.HEAP32[h / 4 + 1];
    let sampleRate = this.Module.HEAP32[h / 4 + 2];

    let samplesPtr = this.Module.HEAP32[h / 4] / 4;
    let samples = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      samples[i] = this.Module.HEAPF32[samplesPtr + i];
    }

    this.Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(h);
    return {samples: samples, sampleRate: sampleRate};
  }
  save(filename, audio) {
    let samples = audio.samples;
    let sampleRate = audio.sampleRate;
    let ptr = this.Module._malloc(samples.length * 4);
    for (let i = 0; i < samples.length; i++) {
      this.Module.HEAPF32[ptr / 4 + i] = samples[i];
    }

    let filenameLen = this.Module.lengthBytesUTF8(filename) + 1;
    let buffer = this.Module._malloc(filenameLen);
    this.Module.stringToUTF8(filename, buffer, filenameLen);
    this.Module._SherpaOnnxWriteWave(ptr, samples.length, sampleRate, buffer);
    this.Module._free(buffer);
    this.Module._free(ptr);
  }
}

function initSherpaOnnxOfflineTts(Module, myConfig) {
  let offlineTtsVitsModelConfig = {
    model: './model.onnx',
    lexicon: '',
    tokens: './tokens.txt',
    dataDir: './espeak-ng-data',
    noiseScale: 0.667,
    noiseScaleW: 0.8,
    lengthScale: 1.0,
  };
  let offlineTtsModelConfig = {
    offlineTtsVitsModelConfig: offlineTtsVitsModelConfig,
    numThreads: 1,
    debug: 1,
    provider: 'cpu',
  };
  let offlineTtsConfig = {
    offlineTtsModelConfig: offlineTtsModelConfig,
    ruleFsts: '',
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
    initSherpaOnnxOfflineTts,
  };
}
