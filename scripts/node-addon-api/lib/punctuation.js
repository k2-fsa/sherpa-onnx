/** @typedef {import('./types').OfflinePunctuationHandle} OfflinePunctuationHandle */
/** @typedef {import('./types').OfflinePunctuationConfig} OfflinePunctuationConfig */
/** @typedef {import('./types').OnlinePunctuationConfig} OnlinePunctuationConfig */

const addon = require('./addon.js');

class OfflinePunctuation {
  /**
   * @param {OfflinePunctuationConfig} config
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
   * @param {OnlinePunctuationConfig} config
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
