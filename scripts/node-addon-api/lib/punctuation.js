/** @typedef {import('./types').OfflinePunctuationHandle} OfflinePunctuationHandle */

const addon = require('./addon.js');

class OfflinePunctuation {
  /**
   * @param {Object} config
   */
  constructor(config) {
    this.handle = addon.createOfflinePunctuation(config);
    this.config = config;
  }
  /**
   * Add punctuation to `text` and return the punctuated text.
   * @param {string} text
   * @returns {string}
   */
  addPunct(text) {
    return addon.offlinePunctuationAddPunct(this.handle, text);
  }
}

class OnlinePunctuation {
  /**
   * @param {Object} config
   */
  constructor(config) {
    this.handle = addon.createOnlinePunctuation(config);
    this.config = config;
  }
  /** @param {string} text @returns {string} */
  addPunct(text) {
    return addon.onlinePunctuationAddPunct(this.handle, text);
  }
}

module.exports = {
  OfflinePunctuation,
  OnlinePunctuation,
} 
