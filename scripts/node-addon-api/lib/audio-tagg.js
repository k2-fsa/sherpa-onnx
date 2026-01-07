/** @typedef {import('./types').AudioTaggingConfig} AudioTaggingConfig */
/** @typedef {import('./types').OfflineStreamObject} OfflineStreamObject */
/** @typedef {import('./types').AudioEvent} AudioEvent */
/** @typedef {import('./types').AudioTaggingHandle} AudioTaggingHandle */

const addon = require('./addon.js');
const non_streaming_asr = require('./non-streaming-asr.js');

/**
 * AudioTagging utility.
 * @class
 */
class AudioTagging {
  /**
   * Create an AudioTagging instance.
   * @param {AudioTaggingConfig} config
   */
  constructor(config) {
    this.handle = addon.createAudioTagging(config);
    this.config = config;
  }

  /**
   * Create an offline stream bound to this AudioTagging instance.
   * @returns {OfflineStreamObject}
   */
  createStream() {
    return new non_streaming_asr.OfflineStream(
        addon.audioTaggingCreateOfflineStream(this.handle));
  }

  /**
   * Compute audio tags from an offline stream.
   * @param {OfflineStreamObject} stream - An offline stream created by `AudioTagging.createStream()`.
   * @param {number} [topK=-1] - Return top K results; -1 for all.
   * @returns {AudioEvent[]}
   */
  compute(stream, topK = -1) {
    return addon.audioTaggingCompute(this.handle, stream.handle, topK);
  }
}

module.exports = {
  AudioTagging,
}
