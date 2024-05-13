const addon = require('./addon.js');

class OfflineStream {
  constructor(handle) {
    this.handle = handle;
  }

  // obj is {samples: samples, sampleRate: sampleRate}
  // samples is a float32 array containing samples in the range [-1, 1]
  // sampleRate is a number
  acceptWaveform(obj) {
    addon.acceptWaveformOffline(this.handle, obj)
  }
}

class OfflineRecognizer {
  constructor(config) {
    this.handle = addon.createOfflineRecognizer(config);
    this.config = config
  }

  createStream() {
    const handle = addon.createOfflineStream(this.handle);
    return new OfflineStream(handle);
  }

  decode(stream) {
    addon.decodeOfflineStream(this.handle, stream.handle);
  }

  getResult(stream) {
    const jsonStr = addon.getOfflineStreamResultAsJson(stream.handle);

    return JSON.parse(jsonStr);
  }
}

module.exports = {
  OfflineRecognizer,
}
