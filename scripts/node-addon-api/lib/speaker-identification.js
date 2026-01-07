/** @typedef {import('./types').SpeakerEmbeddingEntry} SpeakerEmbeddingEntry */
/** @typedef {import('./types').SpeakerEmbeddingManagerAddListFlattenedObj} SpeakerEmbeddingManagerAddListFlattenedObj */
/** @typedef {import('./types').SpeakerEmbeddingManagerSearchObj} SpeakerEmbeddingManagerSearchObj */
/** @typedef {import('./types').SpeakerEmbeddingManagerVerifyObj} SpeakerEmbeddingManagerVerifyObj */
/** @typedef {import('./types').OnlineStreamObject} OnlineStreamObject */

const addon = require('./addon.js');
const streaming_asr = require('./streaming-asr.js');

/**
 * SpeakerEmbeddingExtractor wraps native speaker embedding extractor.
 */
class SpeakerEmbeddingExtractor {
  /**
   * @param {Object} config
   */
  constructor(config) {
    this.handle = addon.createSpeakerEmbeddingExtractor(config);
    this.config = config;
    this.dim = addon.speakerEmbeddingExtractorDim(this.handle);
  }

  /**
   * @returns {OnlineStreamObject}
   */
  createStream() {
    return new streaming_asr.OnlineStream(
        addon.speakerEmbeddingExtractorCreateStream(this.handle));
  }

  /**
   * @param {OnlineStreamObject} stream
   * @returns {boolean}
   */
  isReady(stream) {
    return addon.speakerEmbeddingExtractorIsReady(this.handle, stream.handle);
  }

  /**
   * Compute embedding and return a Float32Array
   * @param {OnlineStreamObject} stream
   * @param {boolean} [enableExternalBuffer=true]
   * @returns {Float32Array}
   */
  compute(stream, enableExternalBuffer = true) {
    return addon.speakerEmbeddingExtractorComputeEmbedding(
        this.handle, stream.handle, enableExternalBuffer);
  }
}

function flatten(arrayList) {
  let n = 0;
  for (let i = 0; i < arrayList.length; ++i) {
    n += arrayList[i].length;
  }
  let ans = new Float32Array(n);

  let offset = 0;
  for (let i = 0; i < arrayList.length; ++i) {
    ans.set(arrayList[i], offset);
    offset += arrayList[i].length;
  }
  return ans;
}

/**
 * Manager for speaker embeddings.
 */
class SpeakerEmbeddingManager {
  constructor(dim) {
    this.handle = addon.createSpeakerEmbeddingManager(dim);
    this.dim = dim;
  }

  /*
   obj = {name: "xxx", v: a-float32-array}
   */
  /**
   * @param {SpeakerEmbeddingEntry} obj
   * @returns {boolean}
   */
  add(obj) {
    return addon.speakerEmbeddingManagerAdd(this.handle, obj);
  }

  /*
   * obj =
   * {name: "xxx", v: [float32_array1, float32_array2, ..., float32_arrayn]
   */
  /**
   * @param {{name:string, v: Float32Array[]}} obj
   * @returns {boolean}
   */
  addMulti(obj) {
    const c = {
      name: obj.name,
      vv: flatten(obj.v),
      n: obj.v.length,
    };
    return addon.speakerEmbeddingManagerAddListFlattened(this.handle, c);
  }

  /**
   * @param {string} name
   * @returns {boolean}
   */
  remove(name) {
    return addon.speakerEmbeddingManagerRemove(this.handle, name);
  }

  /*
   * obj = {v: a-float32-array, threshold: a-float }
   */
  /**
   * @param {SpeakerEmbeddingManagerSearchObj} obj
   * @returns {string}
   */
  search(obj) {
    return addon.speakerEmbeddingManagerSearch(this.handle, obj);
  }

  /*
   * obj = {name: 'xxx', v: a-float32-array, threshold: a-float }
   */
  /**
   * @param {SpeakerEmbeddingManagerVerifyObj} obj
   * @returns {boolean}
   */
  verify(obj) {
    return addon.speakerEmbeddingManagerVerify(this.handle, obj);
  }

  /**
   * @param {string} name
   * @returns {boolean}
   */
  contains(name) {
    return addon.speakerEmbeddingManagerContains(this.handle, name);
  }

  /** @returns {number} */
  getNumSpeakers() {
    return addon.speakerEmbeddingManagerNumSpeakers(this.handle);
  }

  /** @returns {string[]} */
  getAllSpeakerNames() {
    return addon.speakerEmbeddingManagerGetAllSpeakers(this.handle);
  }
}

module.exports = {
  SpeakerEmbeddingExtractor,
  SpeakerEmbeddingManager,
}
