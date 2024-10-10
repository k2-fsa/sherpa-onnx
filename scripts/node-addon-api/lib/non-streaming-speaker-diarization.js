const addon = require('./addon.js');

class OfflineSpeakerDiarization {
  constructor(config) {
    this.handle = addon.createOfflineSpeakerDiarization(config);
    this.config = config;

    this.sampleRate = addon.getOfflineSpeakerDiarizationSampleRate(this.handle);
  }

  /**
   * samples is a 1-d float32 array. Each element of the array should be
   * in the range [-1, 1].
   *
   * We assume its sample rate equals to this.sampleRate.
   *
   * Returns an array of object, where an object is
   *
   *  {
   *    "start": start_time_in_seconds,
   *    "end": end_time_in_seconds,
   *    "speaker": an_integer,
   *  }
   */
  process(samples) {
    return addon.offlineSpeakerDiarizationProcess(this.handle, samples);
  }
}

module.exports = {
  OfflineSpeakerDiarization,
}
