/** @typedef {import('./types').OfflineSpeakerDiarizationConfig} OfflineSpeakerDiarizationConfig */
/** @typedef {import('./types').SpeakerDiarizationSegment} SpeakerDiarizationSegment */

const addon = require('./addon.js');

class OfflineSpeakerDiarization {
  /**
   * @param {OfflineSpeakerDiarizationConfig} config
   */
  constructor(config) {
    this.handle = addon.createOfflineSpeakerDiarization(config);
    this.config = config;

    this.sampleRate = addon.getOfflineSpeakerDiarizationSampleRate(this.handle);
  }

  /**
   * @param {Float32Array} samples - 1-D float32 array in [-1, 1]
   * @returns {SpeakerDiarizationSegment[]}
   */
  process(samples) {
    return addon.offlineSpeakerDiarizationProcess(this.handle, samples);
  }

  /**
   * Set clustering configuration.
   * @param {{clustering: import('./types').FastClusteringConfig}} config
   */
  setConfig(config) {
    addon.offlineSpeakerDiarizationSetConfig(this.handle, config);
    this.config.clustering = config.clustering;
  }
}

module.exports = {
  OfflineSpeakerDiarization,
} 