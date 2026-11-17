import {
  createForcedAligner,
  createForcedAlignerAsync,
  forcedAlignerCreateStream,
  forcedAlignerCompute,
  forcedAlignerComputeAsync,
} from 'libsherpa_onnx.so';
import { OfflineStream } from './NonStreamingAsr';

/**
 * Alignment options of a forced aligner. A stream can override any of them
 * via its own option mechanism before the alignment is computed.
 */
export class ForcedAlignerOptions {
  /**
   * Decoding beam used by the alignment search. Larger is more accurate
   * and slower. If set to 0, the default value 10 is used.
   */
  public beam: number = 10;

  /**
   * Decoding beam used to retry failed alignments. If set to 0, the
   * default value 40 is used.
   */
  public retryBeam: number = 40;

  /**
   * Transition-probability scale relative to the acoustics. If set to 0,
   * the default value 1 is used.
   */
  public transitionScale: number = 1;

  /**
   * Scale of self-loop versus non-self-loop log probs relative to the
   * acoustics. If set to 0, the default value 0.1 is used.
   */
  public selfLoopScale: number = 0.1;

  /**
   * Scaling factor for acoustic likelihoods. If set to 0, the default
   * value 0.1 is used.
   */
  public acousticScale: number = 0.1;
}

/**
 * Configuration of a ForcedAligner.
 */
export class ForcedAlignerConfig {
  /**
   * Directory of one forced-alignment acoustic model. Required files:
   *
   *   - meta.json: feature chain + silence/OOV parameters (parsed once at
   *     construction, field by field)
   *   - final.alimdl: the alignment model (kaldi format: TransitionModel +
   *     AmDiagGmm); note it is final.alimdl, NOT final.mdl
   *   - tree: the context dependency tree (ContextDependency)
   *   - lda.mat: required iff the feature chain is splice+LDA
   *   - phones.txt: optional phone table (falls back to the phones list in
   *     meta.json when missing)
   *
   * Other files of the upstream model bundles (final.mdl, phone_lm.fst,
   * graphemes.txt, rules.yaml, ...) are unused and ignored; missing
   * required files are reported by name. On HarmonyOS the path is
   * relative to the resource root when a resource manager is used.
   */
  public modelDir: string = '';

  /**
   * Path to the pronunciation dictionary used to build the lexicon FST.
   * One pronunciation per line:
   *
   *     key [probability columns ...] phone1 phone2 ...
   *
   * Probability columns (floating point numbers between the key and the
   * phones, e.g. MFA .dict files) are ignored. Several lines under one key
   * give several pronunciations of the same word/character and ALL are
   * kept: each pronunciation becomes a parallel path in the lexicon FST
   * and the acoustics pick the one used by the alignment. This is how
   * Chinese polyphones avoid text-level guessing.
   *
   * The key shape depends on the language: English = words, Chinese = hanzi
   * characters/phrases (hanzi-keyed), other languages = their writing
   * units. Multi-word phrases keep their combined readings: a multi-token
   * key writes one " || "-separated phone group per word/character
   * ("银行 || i n || h a ŋ"); single-token keys need no " || "
   * ("行 x a ŋ"). Words without a pronunciation are reported as
   * out-of-vocabulary and aligned as the model's OOV phone.
   */
  public lexicon: string = '';

  /** Default alignment options; a stream can override any of them before
   * the alignment is computed. */
  public options: ForcedAlignerOptions = new ForcedAlignerOptions();

  /**
   * true to print debug information: model info while loading the model
   * and per-utterance alignment info in compute(). It can also be enabled
   * for a single stream via stream.setOption('debug', '1').
   */
  public debug: boolean = false;
}

/**
 * One aligned interval: a phone in the "phones" tier or a word in the
 * "words" tier.
 */
export class ForcedAlignmentInterval {
  /** Start time in seconds. */
  public start: number = 0;

  /** End time in seconds. */
  public end: number = 0;

  /**
   * The text of this interval: the word for a word interval, the phone
   * symbol for a phone interval. It is empty for a silence gap in the
   * "words" tier (Praat interval tiers must tile the full time range, so
   * the gaps between words are reported as intervals with empty text).
   */
  public text: string = '';
}

/** One tier of the alignment (phones or words). */
export class ForcedAlignmentTier {
  /** Tier name: "phones" or "words"; it is never empty. */
  public name: string = '';

  /** The aligned intervals of this tier, in time order. */
  public intervals: ForcedAlignmentInterval[] = [];
}

/**
 * The result of ForcedAligner.compute(); tiers[0].name == "phones",
 * tiers[1].name == "words".
 */
export class ForcedAlignmentResult {
  /** Duration of the aligned audio in seconds. */
  public duration: number = 0;

  /**
   * The whole alignment as a Praat TextGrid string; save it to a .TextGrid
   * file and open it with Praat (https://www.fon.hum.uva.nl/praat/) to
   * visualize the phone and word boundaries. It is empty for an empty
   * result.
   */
  public textgrid: string = '';

  /** The tiers of the alignment ("phones", "words"); empty for an empty
   * result. */
  public tiers: ForcedAlignmentTier[] = [];
}

/**
 * Forced aligner: given a wave and its transcript, it computes the
 * start/end time of every phone and word.
 *
 *   const config = new ForcedAlignerConfig();
 *   config.modelDir = './english-mfa-v3.1.0';
 *   config.lexicon = './english-mfa-v3.1.0/lexicon.txt';
 *
 *   const aligner = new ForcedAligner(config);
 *   const stream = aligner.createStream();
 *   stream.acceptWaveform(...);
 *   const result = aligner.compute(stream, text);
 */
export class ForcedAligner {
  public config: ForcedAlignerConfig;
  private handle: object;

  /**
   * Create a forced aligner from a model directory and a pronunciation
   * dictionary.
   *
   * @param config configuration for forced alignment
   * @param mgr optional HarmonyOS resource manager used to resolve the
   *   bundled assets of config.modelDir and config.lexicon
   * @param handle optional native handle built earlier; createAsync()
   *   passes it in so the model is not loaded a second time
   */
  constructor(config: ForcedAlignerConfig, mgr?: object, handle?: object) {
    if (handle !== undefined) {
      this.handle = handle;
    } else {
      this.handle = createForcedAligner(config, mgr);
    }
    this.config = config;
  }

  /**
   * Create a forced aligner off the main thread (model loading can take
   * time) and resolve it.
   *
   * @param config configuration for forced alignment
   * @returns the new aligner
   */
  static async createAsync(config: ForcedAlignerConfig): Promise<ForcedAligner> {
    const handle = await createForcedAlignerAsync(config);
    return new ForcedAligner(config, undefined, handle);
  }

  /**
   * Create an utterance stream. Fill it with acceptWaveform(), then pass
   * it to compute().
   */
  createStream(): OfflineStream {
    return new OfflineStream(forcedAlignerCreateStream(this.handle));
  }

  /**
   * Compute the alignment of one utterance.
   *
   * @param stream a stream created by createStream() and filled with a
   *   waveform
   * @param text the transcript of the whole utterance. Hanzi text is
   *   tokenized per character (multi-character dictionary keys give
   *   pronunciation context); anything else is split on whitespace.
   *   Punctuation is ignored
   * @returns the alignment result; empty (no tiers, empty textgrid) with a
   *   warning on stderr when the stream has no features, every word is
   *   out-of-vocabulary, or word boundaries cannot be recovered
   */
  compute(stream: OfflineStream, text: string): ForcedAlignmentResult {
    return forcedAlignerCompute(this.handle, stream.handle, text) as ForcedAlignmentResult;
  }

  /**
   * Compute the alignment of one utterance off the main thread.
   *
   * @param stream a stream created by createStream() and filled with a
   *   waveform
   * @param text the transcript of the whole utterance
   * @returns the alignment result; the same contract as compute()
   */
  async computeAsync(stream: OfflineStream, text: string): Promise<ForcedAlignmentResult> {
    const result = await forcedAlignerComputeAsync(this.handle, stream.handle, text);
    return result as ForcedAlignmentResult;
  }
}
