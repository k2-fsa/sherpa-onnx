const addon = require('./addon.js');

class Display {
  constructor(maxWordPerline) {
    this.handle = addon.createDisplay(maxWordPerline);
  }

  print(idx, text) {
    addon.print(this.handle, idx, text)
  }
}

class OnlineStream {
  constructor(handle) {
    this.handle = handle;
  }

  // obj is {samples: samples, sampleRate: sampleRate}
  // samples is a float32 array containing samples in the range [-1, 1]
  // sampleRate is a number
  acceptWaveform(obj) {
    addon.acceptWaveformOnline(this.handle, obj)
  }

  inputFinished() {
    addon.inputFinished(this.handle)
  }
}

class OnlineRecognizer {
  constructor(config) {
    this.handle = addon.createOnlineRecognizer(config);
    this.config = config
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

  isEndpoint(stream) {
    return addon.isEndpoint(this.handle, stream.handle);
  }

  reset(stream) {
    addon.reset(this.handle, stream.handle);
  }

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
