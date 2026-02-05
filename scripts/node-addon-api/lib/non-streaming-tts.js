/** @typedef {import('./types').OfflineTtsConfig} OfflineTtsConfig */
/** @typedef {import('./types').TtsRequest} TtsRequest */
/** @typedef {import('./types').GeneratedAudio} GeneratedAudio */

const addon = require('./addon.js');

/**
 * Internal symbol to mark async-created TTS instances.
 */
const kFromAsyncFactory = Symbol('OfflineTts.fromAsync');


class GenerationConfig {
  constructor(opts = {}) {
    Object.assign(this, opts)
  }
}


class OfflineTts {
  /**
   * Constructor (sync path).
   *
   * Users call:
   *   new OfflineTts(config)
   *
   * Async factory calls this with an internal descriptor.
   *
   * @param {OfflineTtsConfig|Object} configOrInternal
   */
  constructor(configOrInternal) {
    if (configOrInternal && typeof configOrInternal === 'object' &&
        configOrInternal[kFromAsyncFactory]) {
      // ----- async factory path -----
      this.handle = configOrInternal.handle;
      this.config = configOrInternal.config;
    } else {
      // ----- sync constructor path -----
      this.config = configOrInternal;
      this.handle = addon.createOfflineTts(this.config);
    }

    // Common initialization
    this.numSpeakers = addon.getOfflineTtsNumSpeakers(this.handle);
    this.sampleRate = addon.getOfflineTtsSampleRate(this.handle);
  }

  /**
   * Create an OfflineTts asynchronously (non-blocking).
   * @param {OfflineTtsConfig} config
   * @returns {Promise<OfflineTts>}
   */
  static async createAsync(config) {
    const handle = await addon.createOfflineTtsAsync(config);
    return new OfflineTts({
      [kFromAsyncFactory]: true,
      handle,
      config,
    });
  }

  /**
   * Generate audio synchronously.
   * @param {TtsRequest} obj
   * @returns {GeneratedAudio}
   */
  generate(obj) {
    if (!obj || typeof obj !== 'object') {
      throw new TypeError('generate() expects an object');
    }

    // If generationConfig is present, use new API
    if (obj.generationConfig !== undefined) {
      return addon.offlineTtsGenerateWithConfig(this.handle, obj);
    }

    // Fallback to legacy path
    return addon.offlineTtsGenerate(this.handle, obj);
  }
  /**
   * Asynchronous generation with progress callback
   *
   * The progress callback receives streaming audio chunks.
   *
   * @param {TtsRequest & { onProgress?: (info: { samples: Float32Array,
   *     progress: number }) => number | boolean | void }} obj
   * @returns {Promise<GeneratedAudio>}
   */
  generateAsync(obj) {
    const {onProgress, ...rest} = obj;

    return addon.offlineTtsGenerateAsync(this.handle, {
      ...rest,
      callback: typeof onProgress === 'function' ?
          (info) => {
            const ret = onProgress(info);
            return ret === 0 || ret === false ? 0 : 1;
          } :
          undefined,
    });
  }


  /**
   * Async generation with generationConfig and progress callback
   * @param {TtsRequest & { generationConfig?: object, onProgress?: function }}
   *     obj
   * @returns {Promise<GeneratedAudio>}
   */
  generateAsyncWithConfig(obj) {
    const {onProgress, ...rest} = obj;

    return addon.offlineTtsGenerateAsyncWithConfig(this.handle, {
      ...rest,
      callback: typeof onProgress === 'function' ?
          (info) => {
            const ret = onProgress(info);
            return ret === 0 || ret === false ? 0 : 1;
          } :
          undefined,
    });
  }
}


module.exports = {
  OfflineTts,
  GenerationConfig,
}
