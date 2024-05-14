const addon = require('./addon.js');
const non_streaming_asr = require('./non-streaming-asr.js');

class AudioTagging {
  constructor(config) {
    this.handle = addon.createAudioTagging(config);
    this.config = config;
  }

  createStream() {
    return new non_streaming_asr.OfflineStream(
        addon.audioTaggingCreateOfflineStream(this.handle));
  }

  /* Return an array. Each element is
   * an object {name: "xxx", prob: xxx, index: xxx};
   *
   */
  compute(stream, topK = -1) {
    return addon.audioTaggingCompute(this.handle, stream.handle, topK);
  }
}

module.exports = {
  AudioTagging,
}
