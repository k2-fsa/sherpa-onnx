/** @typedef {import('./types').OfflineSpeechDenoiserConfig} OfflineSpeechDenoiserConfig */
/** @typedef {import('./types').GeneratedAudio} GeneratedAudio */

const addon = require('./addon.js');

class OfflineSpeechDenoiser {
  /**
   * @param {OfflineSpeechDenoiserConfig} config
   */
  constructor(config) {
    this.handle = addon.createOfflineSpeechDenoiser(config);
    this.config = config;

    this.sampleRate =
        addon.offlineSpeechDenoiserGetSampleRateWrapper(this.handle);
  }

  /**
   * Run denoiser synchronously.
   * @param {Object} obj - { samples: Float32Array, sampleRate: number, enableExternalBuffer?: boolean }
   * @returns {GeneratedAudio}
   */
  run(obj) {
    return addon.offlineSpeechDenoiserRunWrapper(this.handle, obj);
  }
}

module.exports = {
  OfflineSpeechDenoiser,
} 