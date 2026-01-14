/** @typedef {import('./types').OfflineStreamObject} OfflineStreamObject */
/** @typedef {import('./types').Waveform} Waveform */
/**
 * @typedef {import('./types').OfflineRecognizerConfig} OfflineRecognizerConfig
 */
/**
 * @typedef {import('./types').OfflineRecognizerResult} OfflineRecognizerResult
 */

const addon = require('./addon.js');

/**
 * OfflineStream represents a synchronous offline audio stream.
 */
class OfflineStream {
  /**
   * @param {OfflineStreamObject|Object} handle - Internal stream object with
   *     `handle` property.
   */
  constructor(handle) {
    this.handle = handle;
  }



  /**
   * Accept a chunk of waveform samples.
   * @param {Waveform} obj - { samples: Float32Array, sampleRate: number }
   */
  acceptWaveform(obj) {
    addon.acceptWaveformOffline(this.handle, obj)
  }
}

/**
 * OfflineRecognizer wraps the native offline recognizer.
 */
class OfflineRecognizer {
  /**
   * Construct a recognizer.
   * @param {OfflineRecognizerConfig|any} configOrHandle
   *   - If OfflineRecognizerConfig: creates a synchronous recognizer.
   *   - If object with { handle, config }: wraps the handle (used by async
   * factory).
   */
  constructor(configOrHandle) {
    if (configOrHandle && configOrHandle.__isNativeHandle) {
      // Wrapping a handle from async creation
      this.handle = configOrHandle.handle;
      this.config = configOrHandle.config;  // save config for reference
    } else if (configOrHandle) {
      // Sync constructor path
      this.handle = addon.createOfflineRecognizer(configOrHandle);
      this.config = configOrHandle;
    } else {
      throw new Error(
          'OfflineRecognizer constructor requires a config or native handle');
    }
  }

  /**
   * Create an OfflineRecognizer asynchronously (non-blocking).
   * @param {OfflineRecognizerConfig} config
   * @returns {Promise<OfflineRecognizer>}
   */
  static async createAsync(config) {
    const handle = await addon.createOfflineRecognizerAsync(config);
    // Wrap handle and config for constructor
    return new OfflineRecognizer({__isNativeHandle: true, handle, config});
  }
  /**
   * Create a new OfflineStream bound to this recognizer.
   * @returns {OfflineStream}
   */
  createStream() {
    const handle = addon.createOfflineStream(this.handle);
    return new OfflineStream(handle);
  }

  /**
   * Replace the recognizer config at runtime.
   * @param {OfflineRecognizerConfig} config
   */
  setConfig(config) {
    addon.offlineRecognizerSetConfig(this.handle, config);
  }

  /**
   * Decode an offline stream (synchronous).
   * @param {OfflineStream} stream
   */
  decode(stream) {
    addon.decodeOfflineStream(this.handle, stream.handle);
  }

  /**
   * Decode an offline stream asynchronously (non-blocking).
   * @param {OfflineStream} stream
   * @returns {Promise<OfflineRecognizerResult>}
   */
  async decodeAsync(stream) {
    const jsonStr =
        await addon.decodeOfflineStreamAsync(this.handle, stream.handle);
    return JSON.parse(jsonStr);
  }

  /**
   * Get recognition result for a stream.
   * @param {OfflineStream} stream
   * @returns {OfflineRecognizerResult}
   */
  getResult(stream) {
    const jsonStr = addon.getOfflineStreamResultAsJson(stream.handle);

    return JSON.parse(jsonStr);
  }
}

module.exports = {
  OfflineRecognizer,
  OfflineStream,
}
