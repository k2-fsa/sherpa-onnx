const addon = require('./addon.js');

class CircularBuffer {
  constructor(capacity) {
    this.handle = addon.createCircularBuffer(capacity);
  }

  // samples is a float32 array
  push(samples) {
    addon.circularBufferPush(this.handle, samples);
  }

  // return a float32 array
  get(startIndex, n, enableExternalBuffer = true) {
    return addon.circularBufferGet(
        this.handle, startIndex, n, enableExternalBuffer);
  }

  pop(n) {
    return addon.circularBufferPop(this.handle, n);
  }

  size() {
    return addon.circularBufferSize(this.handle);
  }

  head() {
    return addon.circularBufferHead(this.handle);
  }

  reset() {
    addon.circularBufferReset(this.handle);
  }
}

class Vad {
  /*
config = {
  sileroVad: {
    model: "./silero_vad.onnx",
    threshold: 0.5,
  }
}
   */
  constructor(config, bufferSizeInSeconds) {
    this.handle =
        addon.createVoiceActivityDetector(config, bufferSizeInSeconds);
    this.config = config;
  }

  acceptWaveform(samples) {
    addon.voiceActivityDetectorAcceptWaveform(this.handle, samples);
  }

  isEmpty() {
    return addon.voiceActivityDetectorIsEmpty(this.handle);
  }

  isDetected() {
    return addon.voiceActivityDetectorIsDetected(this.handle);
  }

  pop() {
    addon.voiceActivityDetectorPop(this.handle);
  }

  clear() {
    addon.voiceActivityDetectorClear(this.handle);
  }

  /*
{
  samples: a 1-d float32 array,
  start: a int32
}
   */
  front(enableExternalBuffer = true) {
    return addon.voiceActivityDetectorFront(this.handle, enableExternalBuffer);
  }

  reset() {
    addon.voiceActivityDetectorReset(this.handle);
  }

  flush() {
    addon.voiceActivityDetectorFlush(this.handle);
  }
}

module.exports = {
  Vad,
  CircularBuffer,
}
