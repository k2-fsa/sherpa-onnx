const addon = require('./addon.js');

class OfflineTts {
  constructor(config) {
    this.handle = addon.createOfflineTts(config);
    this.config = config;

    this.numSpeakers = addon.getOfflineTtsNumSpeakers(this.handle);
    this.sampleRate = addon.getOfflineTtsSampleRate(this.handle);
  }

  /*
   input obj: {text: "xxxx", sid: 0, speed: 1.0}
   where text is a string, sid is a int32, speed is a float

   return an object {samples: Float32Array, sampleRate: <a number>}
   */
  generate(obj) {
    return addon.offlineTtsGenerate(this.handle, obj);
  }
}

module.exports = {
  OfflineTts,
}
