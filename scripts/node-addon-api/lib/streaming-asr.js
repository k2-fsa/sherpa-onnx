/** @typedef {import('./types').OnlineStreamObject} OnlineStreamObject */
/** @typedef {import('./types').OnlineRecognizerHandle} OnlineRecognizerHandle */
/** @typedef {import('./types').OnlineStreamHandle} OnlineStreamHandle */
/** @typedef {import('./types').DisplayHandle} DisplayHandle */
/** @typedef {import('./types').DisplayObject} DisplayObject */
/** @typedef {import('./types').OnlineRecognizerConfig} OnlineRecognizerConfig */
/** @typedef {import('./types').Waveform} Waveform */
/** @typedef {import('./types').OnlineRecognizerResult} OnlineRecognizerResult */

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

  /**
   * Set a string option on the underlying online stream.
   *
   * For example, multilingual streaming Nemotron models read the option
   * 'language' (e.g. 'en', 'de') from each stream on every decode call;
   * leaving it unset selects automatic language detection.
   * @param {string} key
   * @param {string} value
   */
  setOption(key, value) {
    addon.onlineStreamSetOption(this.handle, key, value);
  }

  /**
   * Get a string option of the underlying online stream.
   *
   * Returns an empty string if the option has not been set; use hasOption()
   * to distinguish an unset option from an empty value.
   * @param {string} key
   * @returns {string}
   */
  getOption(key) {
    return addon.onlineStreamGetOption(this.handle, key);
  }

  /**
   * Check whether an option has been set on the underlying online stream.
   * @param {string} key
   * @returns {boolean}
   */
  hasOption(key) {
    return addon.onlineStreamHasOption(this.handle, key);
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
   * Decode multiple streams of this recognizer in parallel.
   *
   * The caller must ensure every stream in the array is ready for decoding,
   * i.e. isReady() returns true for each of them.
   * @param {OnlineStream[]} streams
   */
  decodeStreams(streams) {
    const handles = streams.map((stream) => {
      if (!(stream instanceof OnlineStream)) {
        throw new TypeError('Every element should be an OnlineStream');
      }
      return stream.handle;
    });
    addon.decodeMultipleOnlineStreams(this.handle, handles);
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
