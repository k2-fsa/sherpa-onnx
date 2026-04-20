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

  if ('dpdfnet' in config) {
    freeConfig(config.dpdfnet, Module)
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

function initSherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig(config, Module) {
  if (!('model' in config)) {
    config.model = '';
  }

  const modelLen = Module.lengthBytesUTF8(config.model) + 1;
  const n = modelLen;
  const buffer = Module._malloc(n);
  const len = 1 * 4;
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model, buffer, modelLen);
  Module.setValue(ptr, buffer, 'i8*');

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

  if (!('dpdfnet' in config)) {
    config.dpdfnet = {model: ''};
  }

  const gtcrn =
      initSherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig(config.gtcrn, Module);
  const dpdfnet =
      initSherpaOnnxOfflineSpeechDenoiserDpdfNetModelConfig(
          config.dpdfnet, Module);

  const len = gtcrn.len + 3 * 4 + dpdfnet.len;
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

  Module._CopyHeap(dpdfnet.ptr, dpdfnet.len, ptr + offset);
  offset += dpdfnet.len;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
    gtcrn: gtcrn,
    dpdfnet: dpdfnet,
  };
}

function initSherpaOnnxOfflineSpeechDenoiserConfig(config, Module) {
  if (!('model' in config)) {
    config.model = {
      gtcrn: {model: ''},
      dpdfnet: {model: ''},
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

function copyDenoisedAudio(handle, Module) {
  const numSamples = Module.HEAP32[handle / 4 + 1];
  const denoisedSampleRate = Module.HEAP32[handle / 4 + 2];
  const samplesPtr = Module.HEAP32[handle / 4] / 4;
  const denoisedSamples = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    denoisedSamples[i] = Module.HEAPF32[samplesPtr + i];
  }

  Module._SherpaOnnxDestroyDenoisedAudio(handle);
  return {samples: denoisedSamples, sampleRate: denoisedSampleRate};
}

class SpeechDenoiserBase {
  constructor(Module) {
    this.Module = Module;
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

class OfflineSpeechDenoiser extends SpeechDenoiserBase {
  constructor(configObj, Module) {
    super(Module);
    const config = initSherpaOnnxOfflineSpeechDenoiserConfig(configObj, Module);
    const handle = Module._SherpaOnnxCreateOfflineSpeechDenoiser(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.sampleRate =
        Module._SherpaOnnxOfflineSpeechDenoiserGetSampleRate(this.handle);
  }

  free() {
    this.Module._SherpaOnnxDestroyOfflineSpeechDenoiser(this.handle);
    this.handle = 0;
  }

  run(samples, sampleRate) {
    const pointer =
        this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
    this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
    const h = this.Module._SherpaOnnxOfflineSpeechDenoiserRun(
        this.handle, pointer, samples.length, sampleRate);
    this.Module._free(pointer);

    return copyDenoisedAudio(h, this.Module);
  }
}

class OnlineSpeechDenoiser extends SpeechDenoiserBase {
  constructor(configObj, Module) {
    super(Module);
    const config = initSherpaOnnxOfflineSpeechDenoiserConfig(configObj, Module);
    const handle = Module._SherpaOnnxCreateOnlineSpeechDenoiser(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.sampleRate =
        Module._SherpaOnnxOnlineSpeechDenoiserGetSampleRate(this.handle);
    this.frameShiftInSamples =
        Module._SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(
            this.handle);
  }

  free() {
    this.Module._SherpaOnnxDestroyOnlineSpeechDenoiser(this.handle);
    this.handle = 0;
  }

  run(samples, sampleRate) {
    const pointer =
        this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
    this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
    const h = this.Module._SherpaOnnxOnlineSpeechDenoiserRun(
        this.handle, pointer, samples.length, sampleRate);
    this.Module._free(pointer);

    return copyDenoisedAudio(h, this.Module);
  }

  flush() {
    const h = this.Module._SherpaOnnxOnlineSpeechDenoiserFlush(this.handle);
    return copyDenoisedAudio(h, this.Module);
  }

  reset() {
    this.Module._SherpaOnnxOnlineSpeechDenoiserReset(this.handle);
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

function createOnlineSpeechDenoiser(Module, myConfig) {
  let config = {
    model: {
      gtcrn: {model: './gtcrn.onnx'},
      debug: 0,
    },
  };

  if (myConfig) {
    config = myConfig;
  }

  return new OnlineSpeechDenoiser(config, Module);
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createOfflineSpeechDenoiser,
    createOnlineSpeechDenoiser,
  };
}
