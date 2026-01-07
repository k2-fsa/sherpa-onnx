/** @typedef {import('./types').OnlineStreamObject} OnlineStreamObject */
/** @typedef {import('./types').OnlineRecognizerHandle} OnlineRecognizerHandle */
/** @typedef {import('./types').DisplayObject} DisplayObject */
/** @typedef {import('./types').OnlineRecognizerConfig} OnlineRecognizerConfig */
/** @typedef {import('./types').Waveform} Waveform */

const addon = require('./addon.js');

/**
 * Display helper for printing recognized words.
 */
class Display {
  /**
   * @param {number} maxWordPerline
   */
  constructor(maxWordPerline) {
    this.handle = addon.createDisplay(maxWordPerline);
  }

  /**
   * Print text to display.
   * @param {number} idx
   * @param {string} text
   */
  print(idx, text) {
    addon.print(this.handle, idx, text)
  }
}

/**
 * OnlineStream holds an active online stream handle.
 */
class OnlineStream {
  /**
   * @param {OnlineStreamObject|Object} handle - object with `handle` property
   */
  constructor(handle) {
    this.handle = handle;
  }

  /**
   * Accept waveform data
   * @param {Waveform} obj - { samples: Float32Array, sampleRate: number }
   */
  acceptWaveform(obj) {
    addon.acceptWaveformOnline(this.handle, obj)
  }

  /** Notify the stream input has finished. */
  inputFinished() {
    addon.inputFinished(this.handle)
  }
}

/**
 * OnlineRecognizer wraps native online recognizer.
 */
class OnlineRecognizer {
  /**
   * @param {OnlineRecognizerConfig} config - online recognizer config (see C++ for fields)
   */
  constructor(config) {
    this.handle = addon.createOnlineRecognizer(config);
    this.config = config
  }

  /**
   * Create a new OnlineStream.
   * @returns {OnlineStream}
   */
  createStream() {
    const handle = addon.createOnlineStream(this.handle);
    return new OnlineStream(handle);
  }

  /**
   * Check whether a stream is ready.
   * @param {OnlineStream} stream
   * @returns {boolean}
   */
  isReady(stream) {
    return addon.isOnlineStreamReady(this.handle, stream.handle);
  }

  /**
   * Trigger decoding on a stream.
   * @param {OnlineStream} stream
   */
  decode(stream) {
    addon.decodeOnlineStream(this.handle, stream.handle);
  }

  /**
   * Check endpoint condition for a stream.
   * @param {OnlineStream} stream
   * @returns {boolean}
   */
  isEndpoint(stream) {
    return addon.isEndpoint(this.handle, stream.handle);
  }

  /**
   * Reset a stream.
   * @param {OnlineStream} stream
   */
  reset(stream) {
    addon.reset(this.handle, stream.handle);
  }

  /**
   * Get recognition result for a stream.
   * @param {OnlineStream} stream
   * @returns {OnlineRecognizerResult}
   */
  getResult(stream) {
    const jsonStr =
        addon.getOnlineStreamResultAsJson(this.handle, stream.handle);

    return JSON.parse(jsonStr);
  }
}

module.exports = {
  OnlineRecognizer,
  OnlineStream,
  Display
}
