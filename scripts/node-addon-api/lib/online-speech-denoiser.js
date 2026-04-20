/** @typedef {import('./types').OnlineSpeechDenoiserConfig} OnlineSpeechDenoiserConfig */
/** @typedef {import('./types').GeneratedAudio} GeneratedAudio */
/** @typedef {import('./types').AudioProcessRequest} AudioProcessRequest */

const addon = require('./addon.js');

class OnlineSpeechDenoiser {
  /**
   * @param {OnlineSpeechDenoiserConfig} config
   */
  constructor(config) {
    this.handle = addon.createOnlineSpeechDenoiser(config);
    this.config = config;

    this.sampleRate =
        addon.onlineSpeechDenoiserGetSampleRateWrapper(this.handle);
    this.frameShiftInSamples =
        addon.onlineSpeechDenoiserGetFrameShiftInSamplesWrapper(this.handle);
  }

  /**
   * @param {AudioProcessRequest} obj
   * @returns {GeneratedAudio}
   */
  run(obj) {
    return addon.onlineSpeechDenoiserRunWrapper(this.handle, obj);
  }

  /**
   * @param {boolean} [enableExternalBuffer=true]
   * @returns {GeneratedAudio}
   */
  flush(enableExternalBuffer = true) {
    return addon.onlineSpeechDenoiserFlushWrapper(
        this.handle, enableExternalBuffer);
  }

  reset() {
    addon.onlineSpeechDenoiserResetWrapper(this.handle);
  }
}

module.exports = {
  OnlineSpeechDenoiser,
};
