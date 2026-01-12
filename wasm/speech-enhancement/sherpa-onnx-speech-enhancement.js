function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('config' in config) {
    freeConfig(config.config, Module)
  }

  if ('gtcrn' in config) {
    freeConfig(config.gtcrn, Module)
  }

  Module._free(config.ptr);
}

function initSherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig(config, Module) {
  if (!('model' in config)) {
    config.model = '';
  }

  const modelLen = Module.lengthBytesUTF8(config.model) + 1;

  const n = modelLen;

  const buffer = Module._malloc(n);

  const len = 1 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model, buffer + offset, modelLen);
  offset += modelLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');
  offset += modelLen;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineSpeechDenoiserModelConfig(config, Module) {
  if (!('gtcrn' in config)) {
    config.gtcrn = {model: ''};
  }

  const gtcrn =
      initSherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig(config.gtcrn, Module);

  const len = gtcrn.len + 3 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(gtcrn.ptr, gtcrn.len, ptr + offset);
  offset += gtcrn.len;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.debug || 0, 'i32');
  offset += 4;

  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const buffer = Module._malloc(providerLen);
  Module.stringToUTF8(config.provider || 'cpu', buffer, providerLen);
  Module.setValue(ptr + offset, buffer, 'i8*');
  offset += 4;

  return {buffer: buffer, ptr: ptr, len: len, gtcrn: gtcrn};
}

function initSherpaOnnxOfflineSpeechDenoiserConfig(config, Module) {
  if (!('model' in config)) {
    config.model = {
      gtcrn: {model: ''},
      provider: 'cpu',
      debug: 1,
      numThreads: 1,
    };
  }

  const modelConfig =
      initSherpaOnnxOfflineSpeechDenoiserModelConfig(config.model, Module);
  const len = modelConfig.len;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(modelConfig.ptr, modelConfig.len, ptr + offset);
  offset += modelConfig.len;

  return {
    ptr: ptr,
    len: len,
    config: modelConfig,
  };
}

class OfflineSpeechDenoiser {
  constructor(configObj, Module) {
    console.log(configObj)
    const config = initSherpaOnnxOfflineSpeechDenoiserConfig(configObj, Module)
    // Module._MyPrint(config.ptr);
    const handle = Module._SherpaOnnxCreateOfflineSpeechDenoiser(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.sampleRate =
        Module._SherpaOnnxOfflineSpeechDenoiserGetSampleRate(this.handle);
    this.Module = Module
  }

  free() {
    this.Module._SherpaOnnxDestroyOfflineSpeechDenoiser(this.handle);
    this.handle = 0
  }

  /**
   * @param samples {Float32Array} Containing samples in the range [-1, 1]
   * @param sampleRate {Number}
   */
  run(samples, sampleRate) {
    const pointer =
        this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
    this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
    const h = this.Module._SherpaOnnxOfflineSpeechDenoiserRun(
        this.handle, pointer, samples.length, sampleRate);
    this.Module._free(pointer);

    const numSamples = this.Module.HEAP32[h / 4 + 1];
    const denoisedSampleRate = this.Module.HEAP32[h / 4 + 2];

    const samplesPtr = this.Module.HEAP32[h / 4] / 4;
    const denoisedSamples = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      denoisedSamples[i] = this.Module.HEAPF32[samplesPtr + i];
    }

    this.Module._SherpaOnnxDestroyDenoisedAudio(h);
    return {samples: denoisedSamples, sampleRate: denoisedSampleRate};
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

function createOfflineSpeechDenoiser(Module, myConfig) {
  let config = {
    model: {
      gtcrn: {model: './gtcrn.onnx'},
      debug: 0,
    },
  };

  if (myConfig) {
    config = myConfig;
  }

  return new OfflineSpeechDenoiser(config, Module);
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createOfflineSpeechDenoiser,
  };
}
