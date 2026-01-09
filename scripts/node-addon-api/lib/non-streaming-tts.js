/** @typedef {import('./types').OfflineTtsConfig} OfflineTtsConfig */
/** @typedef {import('./types').TtsRequest} TtsRequest */
/** @typedef {import('./types').GeneratedAudio} GeneratedAudio */

const addon = require('./addon.js');

class OfflineTts {
  /**
   * @param {OfflineTtsConfig} config
   */
  constructor(config) {
    this.handle = addon.createOfflineTts(config);
    this.config = config;

    this.numSpeakers = addon.getOfflineTtsNumSpeakers(this.handle);
    this.sampleRate = addon.getOfflineTtsSampleRate(this.handle);
  }

  /**
   * Generate audio synchronously.
   * @param {TtsRequest} obj
   * @returns {GeneratedAudio}
   */
  generate(obj) {
    return addon.offlineTtsGenerate(this.handle, obj);
  }
}

module.exports = {
  OfflineTts,
} 