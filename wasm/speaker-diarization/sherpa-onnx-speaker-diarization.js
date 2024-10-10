
function freeConfig(config, Module) {
  if ('buffer' in config) {
    Module._free(config.buffer);
  }

  if ('config' in config) {
    freeConfig(config.config, Module)
  }

  if ('segmentation' in config) {
    freeConfig(config.segmentation, Module)
  }

  if ('embedding' in config) {
    freeConfig(config.embedding, Module)
  }

  if ('clustering' in config) {
    freeConfig(config.clustering, Module)
  }

  Module._free(config.ptr);
}

function initSherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(
    config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;
  const n = modelLen;
  const buffer = Module._malloc(n);

  const len = 1 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
  offset += modelLen;

  offset = 0;
  Module.setValue(ptr, buffer + offset, 'i8*');

  return {
    buffer: buffer, ptr: ptr, len: len,
  }
}

function initSherpaOnnxOfflineSpeakerSegmentationModelConfig(config, Module) {
  if (!('pyannote' in config)) {
    config.pyannote = {
      model: '',
    };
  }

  const pyannote = initSherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(
      config.pyannote, Module);

  const len = pyannote.len + 3 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(pyannote.ptr, pyannote.len, ptr + offset);
  offset += pyannote.len;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.debug || 1, 'i32');
  offset += 4;

  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const buffer = Module._malloc(providerLen);
  Module.stringToUTF8(config.provider || 'cpu', buffer, providerLen);
  Module.setValue(ptr + offset, buffer, 'i8*');

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
    config: pyannote,
  };
}

function initSherpaOnnxSpeakerEmbeddingExtractorConfig(config, Module) {
  const modelLen = Module.lengthBytesUTF8(config.model || '') + 1;
  const providerLen = Module.lengthBytesUTF8(config.provider || 'cpu') + 1;
  const n = modelLen + providerLen;
  const buffer = Module._malloc(n);

  const len = 4 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.stringToUTF8(config.model || '', buffer + offset, modelLen);
  offset += modelLen;

  Module.stringToUTF8(config.provider || 'cpu', buffer + offset, providerLen);
  offset += providerLen;

  offset = 0
  Module.setValue(ptr + offset, buffer, 'i8*');
  offset += 4;

  Module.setValue(ptr + offset, config.numThreads || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.debug || 1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, buffer + modelLen, 'i8*');
  offset += 4;

  return {
    buffer: buffer,
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxFastClusteringConfig(config, Module) {
  const len = 2 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module.setValue(ptr + offset, config.numClusters || -1, 'i32');
  offset += 4;

  Module.setValue(ptr + offset, config.threshold || 0.5, 'float');
  offset += 4;

  return {
    ptr: ptr,
    len: len,
  };
}

function initSherpaOnnxOfflineSpeakerDiarizationConfig(config, Module) {
  if (!('segmentation' in config)) {
    config.segmentation = {
      pyannote: {model: ''},
      numThreads: 1,
      debug: 0,
      provider: 'cpu',
    };
  }

  if (!('embedding' in config)) {
    config.embedding = {
      model: '',
      numThreads: 1,
      debug: 0,
      provider: 'cpu',
    };
  }

  if (!('clustering' in config)) {
    config.clustering = {
      numClusters: -1,
      threshold: 0.5,
    };
  }

  const segmentation = initSherpaOnnxOfflineSpeakerSegmentationModelConfig(
      config.segmentation, Module);

  const embedding =
      initSherpaOnnxSpeakerEmbeddingExtractorConfig(config.embedding, Module);

  const clustering =
      initSherpaOnnxFastClusteringConfig(config.clustering, Module);

  const len = segmentation.len + embedding.len + clustering.len + 2 * 4;
  const ptr = Module._malloc(len);

  let offset = 0;
  Module._CopyHeap(segmentation.ptr, segmentation.len, ptr + offset);
  offset += segmentation.len;

  Module._CopyHeap(embedding.ptr, embedding.len, ptr + offset);
  offset += embedding.len;

  Module._CopyHeap(clustering.ptr, clustering.len, ptr + offset);
  offset += clustering.len;

  Module.setValue(ptr + offset, config.minDurationOn || 0.2, 'float');
  offset += 4;

  Module.setValue(ptr + offset, config.minDurationOff || 0.5, 'float');
  offset += 4;

  return {
    ptr: ptr, len: len, segmentation: segmentation, embedding: embedding,
        clustering: clustering,
  }
}

class OfflineSpeakerDiarization {
  constructor(configObj, Module) {
    const config =
        initSherpaOnnxOfflineSpeakerDiarizationConfig(configObj, Module)
    // Module._MyPrint(config.ptr);

    const handle =
        Module._SherpaOnnxCreateOfflineSpeakerDiarization(config.ptr);

    freeConfig(config, Module);

    this.handle = handle;
    this.sampleRate =
        Module._SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(this.handle);
    this.Module = Module

                  this.config = configObj;
  }

  free() {
    this.Module._SherpaOnnxDestroyOfflineSpeakerDiarization(this.handle);
    this.handle = 0
  }

  setConfig(configObj) {
    if (!('clustering' in configObj)) {
      return;
    }

    const config =
        initSherpaOnnxOfflineSpeakerDiarizationConfig(configObj, this.Module);

    this.Module._SherpaOnnxOfflineSpeakerDiarizationSetConfig(
        this.handle, config.ptr);

    freeConfig(config, Module);

    this.config.clustering = configObj.clustering;
  }

  process(samples) {
    const pointer =
        this.Module._malloc(samples.length * samples.BYTES_PER_ELEMENT);
    this.Module.HEAPF32.set(samples, pointer / samples.BYTES_PER_ELEMENT);

    let r = this.Module._SherpaOnnxOfflineSpeakerDiarizationProcess(
        this.handle, pointer, samples.length);
    this.Module._free(pointer);

    let numSegments =
        this.Module._SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(r);

    let segments =
        this.Module._SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
            r);

    let ans = [];

    let sizeOfSegment = 3 * 4;
    for (let i = 0; i < numSegments; ++i) {
      let p = segments + i * sizeOfSegment

      let start = this.Module.HEAPF32[p / 4 + 0];
      let end = this.Module.HEAPF32[p / 4 + 1];
      let speaker = this.Module.HEAP32[p / 4 + 2];

      ans.push({start: start, end: end, speaker: speaker});
    }

    this.Module._SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
    this.Module._SherpaOnnxOfflineSpeakerDiarizationDestroyResult(r);

    return ans;
  }
}

function createOfflineSpeakerDiarization(Module, myConfig) {
  const config = {
    segmentation: {
      pyannote: {model: './segmentation.onnx'},
    },
    embedding: {model: './embedding.onnx'},
    clustering: {numClusters: -1, threshold: 0.5},
    minDurationOn: 0.3,
    minDurationOff: 0.5,
  };

  if (myConfig) {
    config = myConfig;
  }

  return new OfflineSpeakerDiarization(config, Module);
}

if (typeof process == 'object' && typeof process.versions == 'object' &&
    typeof process.versions.node == 'string') {
  module.exports = {
    createOfflineSpeakerDiarization,
  };
}
