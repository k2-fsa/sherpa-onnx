/** @typedef {import('./types').LinearResamplerHandle} LinearResamplerHandle */

const addon = require('./addon.js');

/**
 * A linear resampler that converts audio from one sample rate to another.
 */
class LinearResampler {
  /**
   * Create a linear resampler.
   *
   * @param {number} inputSampleRate - Input sample rate in Hz.
   * @param {number} outputSampleRate - Output sample rate in Hz.
   */
  constructor(inputSampleRate, outputSampleRate) {
    /** @type {LinearResamplerHandle} */
    this.handle =
        addon.createLinearResampler(inputSampleRate, outputSampleRate);
    this.inputSampleRate = inputSampleRate;
    this.outputSampleRate = outputSampleRate;
  }

  /**
   * Resample a chunk of audio samples.
   *
   * Call this for each chunk of input audio. For the final chunk, call
   * {@link flush} instead so that any internally buffered samples are
   * emitted.
   *
   * @param {Float32Array} samples - Input audio samples.
   * @returns {Float32Array} Resampled audio samples.
   */
  resample(samples) {
    return addon.resampleLinear(this.handle, samples, 0);
  }

  /**
   * Resample the final chunk of audio and flush internal buffers.
   *
   * This is the same as {@link resample} but sets flush=1 so that any
   * remaining samples buffered inside the resampler are emitted. Call
   * this once after the last chunk of input audio.
   *
   * @param {Float32Array} samples - The final chunk of input audio samples.
   * @returns {Float32Array} Resampled audio samples including buffered tail.
   */
  flush(samples) {
    return addon.resampleLinear(this.handle, samples, 1);
  }

  /**
   * Reset the resampler to its initial state, discarding any internal
   * buffered samples.
   */
  reset() {
    addon.linearResamplerReset(this.handle);
  }

  /**
   * Get the input sample rate.
   *
   * @returns {number} Input sample rate in Hz.
   */
  getInputSampleRate() {
    return addon.linearResamplerGetInputSampleRate(this.handle);
  }

  /**
   * Get the output sample rate.
   *
   * @returns {number} Output sample rate in Hz.
   */
  getOutputSampleRate() {
    return addon.linearResamplerGetOutputSampleRate(this.handle);
  }
}

module.exports = {
  LinearResampler,
}
