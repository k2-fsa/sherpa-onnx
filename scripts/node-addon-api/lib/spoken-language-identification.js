const addon = require('./addon.js');
const non_streaming_asr = require('./non-streaming-asr.js');

class SpokenLanguageIdentification {
  constructor(config) {
    this.handle = addon.createSpokenLanguageIdentification(config);
    this.config = config;
  }

  createStream() {
    return new non_streaming_asr.OfflineStream(
        addon.createSpokenLanguageIdentificationOfflineStream(this.handle));
  }

  // return a string containing the language code (2 characters),
  // e.g., en, de, fr, es, zh
  // en -> English
  // de -> German
  // fr -> French
  // es -> Spanish
  // zh -> Chinese
  compute(stream) {
    return addon.spokenLanguageIdentificationCompute(
        this.handle, stream.handle);
  }
}

module.exports = {
  SpokenLanguageIdentification,
}
