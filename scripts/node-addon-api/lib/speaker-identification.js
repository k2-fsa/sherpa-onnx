/** @typedef {import('./types').SpeakerEmbeddingEntry} SpeakerEmbeddingEntry */
/** @typedef {import('./types').SpeakerEmbeddingManagerSearchObj} SpeakerEmbeddingManagerSearchObj */
/** @typedef {import('./types').SpeakerEmbeddingManagerVerifyObj} SpeakerEmbeddingManagerVerifyObj */
/** @typedef {import('./types').SpeakerEmbeddingExtractorConfig} SpeakerEmbeddingExtractorConfig */
/** @typedef {import('./streaming-asr').OnlineStream} OnlineStream */

const addon = require('./addon.js');
const streaming_asr = require('./streaming-asr.js');

/**
 * SpeakerEmbeddingExtractor wraps native speaker embedding extractor.
 */
class SpeakerEmbeddingExtractor {
  /**
   * @param {SpeakerEmbeddingExtractorConfig} config
   */
  constructor(config) {
    this.handle = addon.createSpeakerEmbeddingExtractor(config);
    this.config = config;
    this.dim = addon.speakerEmbeddingExtractorDim(this.handle);
  }

  /**
   * @returns {OnlineStream}
   */
  createStream() {
    return new streaming_asr.OnlineStream(
        addon.speakerEmbeddingExtractorCreateStream(this.handle));
  }

  /**
   * @param {OnlineStream} stream
   * @returns {boolean}
   */
  isReady(stream) {
    return addon.speakerEmbeddingExtractorIsReady(this.handle, stream.handle);
  }

  /**
   * Compute embedding and return a Float32Array
   * @param {OnlineStream} stream
   * @param {boolean} [enableExternalBuffer=true]
   * @returns {Float32Array}
   */
  compute(stream, enableExternalBuffer = true) {
    return addon.speakerEmbeddingExtractorComputeEmbedding(
        this.handle, stream.handle, enableExternalBuffer);
  }
}

/**
 * Flattens an array of Float32Arrays into a single Float32Array.
 * @param {Float32Array[]} arrayList
 * @returns {Float32Array}
 */
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
  /**
   * @param {number} dim - The embedding dimension
   */
  constructor(dim) {
    this.handle = addon.createSpeakerEmbeddingManager(dim);
    this.dim = dim;
  }

  /**
   * @param {SpeakerEmbeddingEntry} obj
   * @returns {boolean}
   */
  add(obj) {
    return addon.speakerEmbeddingManagerAdd(this.handle, obj);
  }

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

  /**
   * @param {SpeakerEmbeddingManagerSearchObj} obj
   * @returns {string}
   */
  search(obj) {
    return addon.speakerEmbeddingManagerSearch(this.handle, obj);
  }

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
