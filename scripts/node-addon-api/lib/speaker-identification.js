const addon = require('./addon.js');
const streaming_asr = require('./streaming-asr.js');

class SpeakerEmbeddingExtractor {
  constructor(config) {
    this.handle = addon.createSpeakerEmbeddingExtractor(config);
    this.config = config;
    this.dim = addon.speakerEmbeddingExtractorDim(this.handle);
  }

  createStream() {
    return new streaming_asr.OnlineStream(
        addon.speakerEmbeddingExtractorCreateStream(this.handle));
  }

  isReady(stream) {
    return addon.speakerEmbeddingExtractorIsReady(this.handle, stream.handle);
  }

  // return a float32 array
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

class SpeakerEmbeddingManager {
  constructor(dim) {
    this.handle = addon.createSpeakerEmbeddingManager(dim);
    this.dim = dim;
  }

  /*
   obj = {name: "xxx", v: a-float32-array}
   */
  add(obj) {
    return addon.speakerEmbeddingManagerAdd(this.handle, obj);
  }

  /*
   * obj =
   * {name: "xxx", v: [float32_array1, float32_array2, ..., float32_arrayn]
   */
  addMulti(obj) {
    const c = {
      name: obj.name,
      vv: flatten(obj.v),
      n: obj.v.length,
    };
    return addon.speakerEmbeddingManagerAddListFlattened(this.handle, c);
  }

  remove(name) {
    return addon.speakerEmbeddingManagerRemove(this.handle, name);
  }

  /*
   * obj = {v: a-float32-array, threshold: a-float }
   */
  search(obj) {
    return addon.speakerEmbeddingManagerSearch(this.handle, obj);
  }

  /*
   * obj = {name: 'xxx', v: a-float32-array, threshold: a-float }
   */
  verify(obj) {
    return addon.speakerEmbeddingManagerVerify(this.handle, obj);
  }

  contains(name) {
    return addon.speakerEmbeddingManagerContains(this.handle, name);
  }

  getNumSpeakers() {
    return addon.speakerEmbeddingManagerNumSpeakers(this.handle);
  }

  getAllSpeakerNames() {
    return addon.speakerEmbeddingManagerGetAllSpeakers(this.handle);
  }
}

module.exports = {
  SpeakerEmbeddingExtractor,
  SpeakerEmbeddingManager,
}
