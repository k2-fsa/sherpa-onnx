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
  /**
   * Asynchronous generation with progress callback
   *
   * The progress callback receives streaming audio chunks.
   *
   * @param {TtsRequest & {
   *   /**
   *    * Optional progress callback called multiple times with partial audio
   *    * @param {{ samples: Float32Array, progress: number }} info
   *    * `progress` in [0,1]
   *    * Return `0` or `false` to cancel, anything else to continue
   *    *\/
   *   onProgress?: (info: { samples: Float32Array, progress: number }) =>
   * number | boolean | void
   * }} obj
   * @returns {Promise<GeneratedAudio>}
   */
  generateAsync(obj) {
    const {onProgress, ...rest} = obj;

    return addon.offlineTtsGenerateAsync(this.handle, {
      ...rest,
      callback: typeof onProgress === 'function' ?
          (info) => {
            // JS contract: 0 or false = cancel, else continue
            const ret = onProgress(info);
            return ret === 0 || ret === false ? 0 : 1;
          } :
          undefined,
    });
  }
}


module.exports = {
  OfflineTts,
}
