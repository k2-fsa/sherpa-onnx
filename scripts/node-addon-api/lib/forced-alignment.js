/** @typedef {import('./types').ForcedAlignerConfig} ForcedAlignerConfig */
/** @typedef {import('./types').ForcedAlignerHandle} ForcedAlignerHandle */
/** @typedef {import('./types').ForcedAlignmentResult} ForcedAlignmentResult */
/** @typedef {import('./types').OfflineStreamHandle} OfflineStreamHandle */

const addon = require('./addon.js');
const {OfflineStream} = require('./non-streaming-asr.js');

/**
 * Private marker used by the async factory below.
 *
 * Not accessible unless someone has a reference to this Symbol.
 */
const kFromAsyncFactory = Symbol('ForcedAligner.fromAsync');

/**
 * ForcedAligner: given a wave and its transcript, it computes the
 * start/end time of every phone and word.
 */
class ForcedAligner {
  /**
   * Create a forced aligner from a model directory and a pronunciation
   * dictionary.
   *
   *   const aligner = new ForcedAligner(config)
   *
   * Async factory calls this with an internal descriptor.
   *
   * @param {ForcedAlignerConfig | Object} configOrInternal
   */
  constructor(configOrInternal) {
    // ----- async factory path -----
    if (configOrInternal && typeof configOrInternal === 'object' &&
        configOrInternal[kFromAsyncFactory]) {
      this.handle = configOrInternal.handle;
      this.config = configOrInternal.config;
      return;
    }

    // ----- sync constructor path -----
    this.handle = addon.createForcedAligner(configOrInternal);
    if (!this.handle) {
      // The C API returns NULL on a bad config (e.g. missing modelDir or
      // lexicon).
      throw new Error('Failed to create the forced aligner');
    }
    this.config = configOrInternal;
  }

  /**
   * Create a forced aligner off the main thread and resolve it.
   * @param {ForcedAlignerConfig} config
   * @returns {Promise<ForcedAligner>}
   */
  static async createAsync(config) {
    const handle = await addon.createForcedAlignerAsync(config);
    return new ForcedAligner({[kFromAsyncFactory]: true, handle, config});
  }

  /**
   * Create an utterance stream. Fill it with acceptWaveform(), then pass
   * it to compute().
   * @returns {OfflineStream}
   */
  createStream() {
    return new OfflineStream(addon.forcedAlignerCreateStream(this.handle));
  }

  /**
   * Compute the alignment of one utterance.
   *
   * @param {OfflineStream} stream
   * @param {string} text - The transcript of the whole utterance. Hanzi
   *   text is tokenized per character (multi-character dictionary keys
   *   give pronunciation context); anything else is split on whitespace.
   *   Punctuation is ignored.
   * @returns {ForcedAlignmentResult} The alignment result; empty (no
   *   tiers, empty textgrid) with a warning on stderr when the stream has
   *   no features, every word is out-of-vocabulary, or word boundaries
   *   cannot be recovered. An interval has an empty text for a silence gap
   *   in the "words" tier.
   */
  compute(stream, text) {
    return addon.forcedAlignerCompute(this.handle, stream.handle, text);
  }

  /**
   * Compute the alignment of one utterance off the main thread.
   * @param {OfflineStream} stream
   * @param {string} text
   * @returns {Promise<ForcedAlignmentResult>}
   */
  async computeAsync(stream, text) {
    return addon.forcedAlignerComputeAsync(this.handle, stream.handle, text);
  }
}

module.exports = {
  ForcedAligner,
};
