function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('sileroVad' in config) {
    freeConfig(config.sileroVad, Module)
  }

  if ('tenVad' in config) {
    freeConfig(config.tenVad, Module)
  }


  Module._free(config.ptr);
}

// The user should free the returned pointers
function initSherpaOnnxSileroVadModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;

  const n = modelLen;

  const buffer = Module._malloc(n);

  const len = 6 * 4;
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, modelLen);

  offset = 0;
  Module.setValue(ptr, buffer, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.threshold || 0.5, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.minSilenceDuration || 0.5, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.minSpeechDuration || 0.25, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.windowSize || 512, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.maxSpeechDuration || 20, 'float');
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxTenVadModelConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;

  const n = modelLen;

  const buffer = Module._malloc(n);

  const len = 6 * 4;
  const ptr = Module._malloc(len);

  Module.stringToUTF8(config.model || '', buffer, modelLen);

  offset = 0;
  Module.setValue(ptr, buffer, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.threshold || 0.5, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.minSilenceDuration || 0.5, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.minSpeechDuration || 0.25, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.windowSize || 256, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.maxSpeechDuration || 20, 'float');
  offset += 4;

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxVadModelConfig(config, Module) {
  if (!('sileroVad' in config)) {
    config.sileroVad = {
      model: '',
      threshold: 0.50,
      minSilenceDuration: 0.50,
      minSpeechDuration: 0.25,
      windowSize: 512,
      maxSpeechDuration: 20,
    };
  }

  if (!('tenVad' in config)) {
    config.tenVad = {
      model: '',
      threshold: 0.50,
      minSilenceDuration: 0.50,
      minSpeechDuration: 0.25,
      windowSize: 256,
      maxSpeechDuration: 20,
    };
  }

  const sileroVad =
      initSherpaOnnxSileroVadModelConfig(config.sileroVad, Module);

  const tenVad = initSherpaOnnxTenVadModelConfig(config.tenVad, Module);

  const len = sileroVad.len + 4 * 4 + tenVad.len;
  const ptr = Module._malloc(len);

  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const buffer = Module._malloc(providerLen);
  Module.stringToUTF8(config.provider || 'cpu', buffer, providerLen);

  let offset = 0;
  Module._CopyHeap(sileroVad.ptr, sileroVad.len, ptr + offset);
  offset += sileroVad.len;

  Module.setValue(ptr + offset, config.sampleRate || 16000, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer, 'i8*');  // provider
  offset += 4;

  Module.setValue(ptr + offset, config.debug || 0, 'i32');
  offset += 4;

  Module._CopyHeap(tenVad.ptr, tenVad.len, ptr + offset);
  offset += tenVad.len;

  return {
    buffer: buffer, ptr: ptr, len: len, sileroVad: sileroVad, tenVad: tenVad
  }
}

function createVad(Module, myConfig) {
  const sileroVad = {
    model: './silero_vad.onnx',
    threshold: 0.50,
    minSilenceDuration: 0.50,
    minSpeechDuration: 0.25,
    maxSpeechDuration: 20,
    windowSize: 512,
  };

  const tenVad = {
    model: '',
    threshold: 0.50,
    minSilenceDuration: 0.50,
    minSpeechDuration: 0.25,
    maxSpeechDuration: 20,
    windowSize: 256,
  };

  let config = {
    sileroVad: sileroVad,
    tenVad: tenVad,
    sampleRate: 16000,
    numThreads: 1,
    provider: 'cpu',
    debug: 1,
    bufferSizeInSeconds: 30,
  };

  if (myConfig) {
    config = myConfig;
  }

  return new Vad(config, Module);
}


class CircularBuffer {
  constructor(capacity, Module) {
    this.handle = Module._SherpaOnnxCreateCircularBuffer(capacity);
    this.Module = Module;
  }

  free() {
    this.Module._SherpaOnnxDestroyCircularBuffer(this.handle);
    this.handle = 0
  }

  /**
   * @param samples {Float32Array}
   */
  push(samples) {
    const pointer =
        this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
    this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
    this.Module._SherpaOnnxCircularBufferPush(
        this.handle, pointer, samples.length);
    this.Module._free(pointer);
  }

  get(startIndex, n) {
    const p =
        this.Module._SherpaOnnxCircularBufferGet(this.handle, startIndex, n);

    const samplesPtr = p / 4;
    const samples = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      samples[i] = this.Module.HEAPF32[samplesPtr + i];
    }

    this.Module._SherpaOnnxCircularBufferFree(p);

    return samples;
  }

  pop(n) {
    this.Module._SherpaOnnxCircularBufferPop(this.handle, n);
  }

  size() {
    return this.Module._SherpaOnnxCircularBufferSize(this.handle);
  }

  head() {
    return this.Module._SherpaOnnxCircularBufferHead(this.handle);
  }

  reset() {
    this.Module._SherpaOnnxCircularBufferReset(this.handle);
  }
}

class Vad {
  constructor(configObj, Module) {
    this.config = configObj;
    const config = initSherpaOnnxVadModelConfig(configObj, Module);
    const handle = Module._SherpaOnnxCreateVoiceActivityDetector(
        config.ptr, configObj.bufferSizeInSeconds || 30);
    freeConfig(config, Module);

    this.handle = handle;
    this.Module = Module;
  }

  free() {
    this.Module._SherpaOnnxDestroyVoiceActivityDetector(this.handle);
    this.handle = 0
  }

  // samples is a float32 array
  acceptWaveform(samples) {
    const pointer =
        this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
    this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);
    this.Module._SherpaOnnxVoiceActivityDetectorAcceptWaveform(
        this.handle, pointer, samples.length);
    this.Module._free(pointer);
  }

  isEmpty() {
    return this.Module._SherpaOnnxVoiceActivityDetectorEmpty(this.handle) == 1;
  }

  isDetected() {
    return this.Module._SherpaOnnxVoiceActivityDetectorDetected(this.handle) ==
        1;
  }

  pop() {
    this.Module._SherpaOnnxVoiceActivityDetectorPop(this.handle);
  }

  clear() {
    this.Module._SherpaOnnxVoiceActivityDetectorClear(this.handle);
  }

  /*
{
  samples: a 1-d float32 array,
  start: an int32
}
   */
  front() {
    const h = this.Module._SherpaOnnxVoiceActivityDetectorFront(this.handle);

    const start = this.Module.HEAP32[h / 4];
    const samplesPtr = this.Module.HEAP32[h / 4 + 1] / 4;
    const numSamples = this.Module.HEAP32[h / 4 + 2];

    const samples = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      samples[i] = this.Module.HEAPF32[samplesPtr + i];
    }

    this.Module._SherpaOnnxDestroySpeechSegment(h);
    return {samples: samples, start: start};
  }

  reset() {
    this.Module._SherpaOnnxVoiceActivityDetectorReset(this.handle);
  }

  flush() {
    this.Module._SherpaOnnxVoiceActivityDetectorFlush(this.handle);
  }
};

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createVad,
    CircularBuffer,
  };
}
