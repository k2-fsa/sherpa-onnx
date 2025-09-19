const addon = require('./addon.js');

class OfflinePunctuation {
  constructor(config) {
    this.handle = addon.createOfflinePunctuation(config);
    this.config = config;
  }
  addPunct(text) {
    return addon.offlinePunctuationAddPunct(this.handle, text);
  }
}

class OnlinePunctuation {
  constructor(config) {
    this.handle = addon.createOnlinePunctuation(config);
    this.config = config;
  }
  addPunct(text) {
    return addon.onlinePunctuationAddPunct(this.handle, text);
  }
}

module.exports = {
  OfflinePunctuation,
  OnlinePunctuation,
}
