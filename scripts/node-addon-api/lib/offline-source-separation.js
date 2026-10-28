/** @typedef {import('./types').OfflineSourceSeparationConfig} OfflineSourceSeparationConfig */
/** @typedef {import('./types').OfflineSourceSeparationHandle} OfflineSourceSeparationHandle */
/** @typedef {import('./types').SourceSeparationRequest} SourceSeparationRequest */
/** @typedef {import('./types').SourceSeparationOutput} SourceSeparationOutput */

const addon = require('./addon.js');

class OfflineSourceSeparation {
  /**
   * @param {OfflineSourceSeparationConfig} config
   */
  constructor(config) {
    /** @type {OfflineSourceSeparationHandle} */
    this.handle = addon.createOfflineSourceSeparation(config);
    this.config = config;
    this.sampleRate =
        addon.offlineSourceSeparationGetOutputSampleRate(this.handle);
    this.numStems =
        addon.offlineSourceSeparationGetNumberOfStems(this.handle);
  }

  /**
   * Separate audio synchronously. Mono input is duplicated to stereo.
   * @param {SourceSeparationRequest} obj
   * @returns {SourceSeparationOutput}
   */
  process(obj) {
    return addon.offlineSourceSeparationProcess(this.handle, obj);
  }
}

module.exports = {
  OfflineSourceSeparation,
};
