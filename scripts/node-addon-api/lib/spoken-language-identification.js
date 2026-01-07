/** @typedef {import('./types').OfflineStreamObject} OfflineStreamObject */
/** @typedef {import('./types').SpokenLanguageIdentificationConfig} SpokenLanguageIdentificationConfig */

const addon = require('./addon.js');
const non_streaming_asr = require('./non-streaming-asr.js');

class SpokenLanguageIdentification {
  /**
   * @param {SpokenLanguageIdentificationConfig} config
   */
  constructor(config) {
    this.handle = addon.createSpokenLanguageIdentification(config);
    this.config = config;
  }

  /**
   * @returns {OfflineStreamObject}
   */
  createStream() {
    return new non_streaming_asr.OfflineStream(
        addon.createSpokenLanguageIdentificationOfflineStream(this.handle));
  }

  /**
   * Return a 2-letter language code, e.g. 'en', 'de', 'fr', 'es', 'zh'
   * @param {OfflineStreamObject} stream
   * @returns {string}
   */
  compute(stream) {
    return addon.spokenLanguageIdentificationCompute(
        this.handle, stream.handle);
  }
}

module.exports = {
  SpokenLanguageIdentification,
} 