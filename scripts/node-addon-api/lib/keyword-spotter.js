const addon = require('./addon.js');
const streaming_asr = require('./streaming-asr.js');

class KeywordSpotter {
  constructor(config) {
    this.handle = addon.createKeywordSpotter(config);
    this.config = config
  }

  createStream() {
    const handle = addon.createKeywordStream(this.handle);
    return new streaming_asr.OnlineStream(handle);
  }

  isReady(stream) {
    return addon.isKeywordStreamReady(this.handle, stream.handle);
  }

  decode(stream) {
    addon.decodeKeywordStream(this.handle, stream.handle);
  }

  getResult(stream) {
    const jsonStr = addon.getKeywordResultAsJson(this.handle, stream.handle);

    return JSON.parse(jsonStr);
  }
}

module.exports = {
  KeywordSpotter,
}
