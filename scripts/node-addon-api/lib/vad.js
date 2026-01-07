/** @typedef {import('./types').CircularBufferHandle} CircularBufferHandle */
/** @typedef {import('./types').SpeechSegment} SpeechSegment */
/** @typedef {import('./types').VadConfig} VadConfig */

const addon = require('./addon.js');

/**
 * CircularBuffer stores float32 samples internally.
 */
class CircularBuffer {
  /**
   * @param {number} capacity - capacity in samples (integer)
   */
  constructor(capacity) {
    this.handle = addon.createCircularBuffer(capacity);
  }

  /**
   * Push samples into the buffer.
   * @param {Float32Array} samples
   */
  push(samples) {
    addon.circularBufferPush(this.handle, samples);
  }

  /**
   * Get a slice of samples.
   * @param {number} startIndex
   * @param {number} n
   * @param {boolean} [enableExternalBuffer=true]
   * @returns {Float32Array}
   */
  get(startIndex, n, enableExternalBuffer = true) {
    return addon.circularBufferGet(
        this.handle, startIndex, n, enableExternalBuffer);
  }

  /**
   * Pop n samples from the buffer.
   * @param {number} n
   */
  pop(n) {
    return addon.circularBufferPop(this.handle, n);
  }

  /**
   * Get current size in samples.
   * @returns {number}
   */
  size() {
    return addon.circularBufferSize(this.handle);
  }

  /**
   * Get head index.
   * @returns {number}
   */
  head() {
    return addon.circularBufferHead(this.handle);
  }

  /** Reset the buffer. */
  reset() {
    addon.circularBufferReset(this.handle);
  }
}

/**
 * Voice Activity Detector (VAD).
 */
class Vad {
  /**
   * @param {VadConfig} config
   * @param {number} bufferSizeInSeconds
   */
  constructor(config, bufferSizeInSeconds) {
    this.handle =
        addon.createVoiceActivityDetector(config, bufferSizeInSeconds);
    this.config = config;
  }

  /**
   * Accept raw waveform samples.
   * @param {Float32Array} samples
   */
  acceptWaveform(samples) {
    addon.voiceActivityDetectorAcceptWaveform(this.handle, samples);
  }

  /** @returns {boolean} */
  isEmpty() {
    return addon.voiceActivityDetectorIsEmpty(this.handle);
  }

  /** @returns {boolean} */
  isDetected() {
    return addon.voiceActivityDetectorIsDetected(this.handle);
  }

  /** Pop the earliest detected speech segment. */
  pop() {
    addon.voiceActivityDetectorPop(this.handle);
  }

  /** Clear internal state. */
  clear() {
    addon.voiceActivityDetectorClear(this.handle);
  }

  /**
   * Get the front speech segment.
   * @param {boolean} [enableExternalBuffer=true]
   * @returns {SpeechSegment}
   */
  front(enableExternalBuffer = true) {
    return addon.voiceActivityDetectorFront(this.handle, enableExternalBuffer);
  }

  /** Reset detector state. */
  reset() {
    addon.voiceActivityDetectorReset(this.handle);
  }

  /** Flush pending internal buffer. */
  flush() {
    addon.voiceActivityDetectorFlush(this.handle);
  }
}

module.exports = {
  Vad,
  CircularBuffer,
}
