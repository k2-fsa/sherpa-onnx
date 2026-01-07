/** @typedef {import('./types').KeywordSpotterConfig} KeywordSpotterConfig */
/** @typedef {import('./types').OnlineStreamObject} OnlineStreamObject */
/** @typedef {import('./types').KeywordResult} KeywordResult */

const addon = require('./addon.js');
const streaming_asr = require('./streaming-asr.js');

/**
 * KeywordSpotter handles keyword detection.
 */
class KeywordSpotter {
  /**
   * @param {KeywordSpotterConfig} config
   */
  constructor(config) {
    this.handle = addon.createKeywordSpotter(config);
    this.config = config
  }

  /**
   * Create an OnlineStream for the spotter.
   * @returns {OnlineStreamObject}
   */
  createStream() {
    const handle = addon.createKeywordStream(this.handle);
    return new streaming_asr.OnlineStream(handle);
  }

  /**
   * @param {OnlineStreamObject} stream
   * @returns {boolean}
   */
  isReady(stream) {
    return addon.isKeywordStreamReady(this.handle, stream.handle);
  }

  /**
   * Trigger decode on a stream.
   * @param {OnlineStreamObject} stream
   */
  decode(stream) {
    addon.decodeKeywordStream(this.handle, stream.handle);
  }

  /**
   * Reset a stream.
   * @param {OnlineStreamObject} stream
   */
  reset(stream) {
    addon.resetKeywordStream(this.handle, stream.handle);
  }

  /**
   * Get the keyword result for a stream.
   * @param {OnlineStreamObject} stream
   * @returns {KeywordResult}
   */
  getResult(stream) {
    const jsonStr = addon.getKeywordResultAsJson(this.handle, stream.handle);

    return JSON.parse(jsonStr);
  }
} 

module.exports = {
  KeywordSpotter,
}
