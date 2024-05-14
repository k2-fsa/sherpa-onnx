const addon = require('./addon.js');

class Punctuation {
  constructor(config) {
    this.handle = addon.createOfflinePunctuation(config);
    this.config = config;
  }
  addPunct(text) {
    return addon.offlinePunctuationAddPunct(this.handle, text);
  }
}

module.exports = {
  Punctuation,
}
