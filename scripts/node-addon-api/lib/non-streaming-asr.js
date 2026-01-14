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
   * @param {OfflineRecognizerConfig} config
   */
  constructor(config) {
    this.handle = addon.createOfflineRecognizer(config);
    this.config = config
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
