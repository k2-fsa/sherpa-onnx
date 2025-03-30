const addon = require('./addon.js');

class OfflineSpeechDenoiser {
  constructor(config) {
    this.handle = addon.createOfflineSpeechDenoiser(config);
    this.config = config;

    this.sampleRate =
        addon.offlineSpeechDenoiserGetSampleRateWrapper(this.handle);
  }

  /*
    obj is
    {samples: samples, sampleRate: sampleRate, enableExternalBuffer: true}

    samples is a float32 array containing samples in the range [-1, 1]
    sampleRate is a number

   return an object {samples: Float32Array, sampleRate: <a number>}
   */
  run(obj) {
    return addon.offlineSpeechDenoiserRunWrapper(this.handle, obj);
  }
}

module.exports = {
  OfflineSpeechDenoiser,
}
