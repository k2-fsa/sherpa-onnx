const addon = require('./addon.js');

class OnlineStream {
  constructor(handle) {
    this.handle = handle;
  }

  // samples is a float32 array containing samples in the range [-1, 1]
  acceptWaveform(samples, sampleRate) {
    addon.acceptWaveformOnline(
        this.handle, {samples: samples, sampleRate: sampleRate})
  }
}

class OnlineRecognizer {
  constructor(config) {
    this.handle = addon.createOnlineRecognizer(config);
  }

  createStream() {
    const handle = addon.createOnlineStream(this.handle);
    return new OnlineStream(handle);
  }

  isReady(stream) {
    return addon.isOnlineStreamReady(this.handle, stream.handle);
  }

  decode(stream) {
    addon.decodeOnlineStream(this.handle, stream.handle);
  }

  getResult(stream) {
    const jsonStr =
        addon.getOnlineStreamResultAsJson(this.handle, stream.handle);

    return JSON.parse(jsonStr);
  }
}

module.exports = {OnlineRecognizer}
