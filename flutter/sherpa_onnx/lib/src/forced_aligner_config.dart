// Copyright (c)  2026  Xiaomi Corporation
// Shared config/data classes for forced alignment -- no FFI, works on all
// platforms.

/// Alignment options of a forced aligner.
///
/// A stream can override any of them before the alignment is computed. The
/// defaults are the C API defaults.
class ForcedAlignmentOptions {
  const ForcedAlignmentOptions({
    this.beam = 10,
    this.retryBeam = 40,
    this.transitionScale = 1,
    this.selfLoopScale = 0.1,
    this.acousticScale = 0.1,
  });

  factory ForcedAlignmentOptions.fromJson(Map<String, dynamic> json) {
    return ForcedAlignmentOptions(
      beam: (json['beam'] as num?)?.toDouble() ?? 10,
      retryBeam: (json['retryBeam'] as num?)?.toDouble() ?? 40,
      transitionScale: (json['transitionScale'] as num?)?.toDouble() ?? 1,
      selfLoopScale: (json['selfLoopScale'] as num?)?.toDouble() ?? 0.1,
      acousticScale: (json['acousticScale'] as num?)?.toDouble() ?? 0.1,
    );
  }

  @override
  String toString() {
    return 'ForcedAlignmentOptions(beam: $beam, retryBeam: $retryBeam, transitionScale: $transitionScale, selfLoopScale: $selfLoopScale, acousticScale: $acousticScale)';
  }

  Map<String, dynamic> toJson() => {
        'beam': beam,
        'retryBeam': retryBeam,
        'transitionScale': transitionScale,
        'selfLoopScale': selfLoopScale,
        'acousticScale': acousticScale,
      };

  /// Decoding beam used by the alignment search. Larger is more accurate
  /// and slower. Default: 10.
  final double beam;

  /// Decoding beam used to retry failed alignments. Default: 40.
  final double retryBeam;

  /// Transition-probability scale relative to the acoustics. Default: 1.
  final double transitionScale;

  /// Scale of self-loop versus non-self-loop log probs relative to the
  /// acoustics. Default: 0.1.
  final double selfLoopScale;

  /// Scaling factor for acoustic likelihoods. Default: 0.1.
  final double acousticScale;
}

/// Configuration of a [ForcedAligner].
class ForcedAlignerConfig {
  const ForcedAlignerConfig({
    required this.modelDir,
    required this.lexicon,
    this.options = const ForcedAlignmentOptions(),
    this.debug = false,
  });

  factory ForcedAlignerConfig.fromJson(Map<String, dynamic> json) {
    return ForcedAlignerConfig(
      modelDir: json['modelDir'] as String,
      lexicon: json['lexicon'] as String,
      options: json['options'] == null
          ? const ForcedAlignmentOptions()
          : ForcedAlignmentOptions.fromJson(
              json['options'] as Map<String, dynamic>),
      debug: json['debug'] as bool? ?? false,
    );
  }

  @override
  String toString() {
    return 'ForcedAlignerConfig(modelDir: $modelDir, lexicon: $lexicon, options: $options, debug: $debug)';
  }

  Map<String, dynamic> toJson() => {
        'modelDir': modelDir,
        'lexicon': lexicon,
        'options': options.toJson(),
        'debug': debug,
      };

  /// Directory of one forced-alignment acoustic model. Required files:
  ///
  /// - meta.json: feature chain + silence/OOV parameters (parsed once at
  ///   construction, field by field)
  /// - final.alimdl: the alignment model (kaldi format: TransitionModel +
  ///   AmDiagGmm); note it is final.alimdl, NOT final.mdl
  /// - tree: the context dependency tree
  /// - lda.mat: required iff the feature chain is splice+LDA
  /// - phones.txt: optional phone table (falls back to the phones list in
  ///   meta.json when missing)
  ///
  /// Other files of the upstream model bundles are unused and ignored;
  /// missing required files are reported by name.
  final String modelDir;

  /// Path to the pronunciation dictionary used to build the lexicon FST.
  /// One pronunciation per line:
  ///
  ///     key [probability columns ...] phone1 phone2 ...
  ///
  /// Probability columns (floating point numbers between the key and the
  /// phones, e.g. MFA .dict files) are ignored. Several lines under one key
  /// give several pronunciations of the same word/character and ALL are
  /// kept: each pronunciation becomes a parallel path in the lexicon FST
  /// and the acoustics pick the one used by the alignment. This is how
  /// Chinese polyphones avoid text-level guessing.
  ///
  /// The key shape depends on the language: English = words, Chinese =
  /// hanzi characters/phrases (hanzi-keyed), other languages = their
  /// writing units. Multi-word phrases keep their combined readings: a
  /// multi-token key writes one " || "-separated phone group per
  /// word/character ("银行 || i n || h a ŋ"); single-token keys need no
  /// " || " ("行 x a ŋ"). Words without a pronunciation are reported as
  /// out-of-vocabulary and aligned as the model's OOV phone.
  final String lexicon;

  /// Default alignment options; a stream can override any of them before
  /// the alignment is computed.
  final ForcedAlignmentOptions options;

  /// true to print debug information: model info while loading the model
  /// and per-utterance alignment info in compute().
  final bool debug;
}

/// One aligned interval: a phone in the "phones" tier or a word in the
/// "words" tier.
class AlignmentInterval {
  const AlignmentInterval({
    required this.start,
    required this.end,
    required this.text,
  });

  factory AlignmentInterval.fromJson(Map<String, dynamic> json) {
    return AlignmentInterval(
      start: (json['start'] as num).toDouble(),
      end: (json['end'] as num).toDouble(),
      text: json['text'] as String,
    );
  }

  @override
  String toString() {
    return 'AlignmentInterval(start: $start, end: $end, text: $text)';
  }

  Map<String, dynamic> toJson() => {
        'start': start,
        'end': end,
        'text': text,
      };

  /// Start time in seconds.
  final double start;

  /// End time in seconds.
  final double end;

  /// The label of this interval: the word for a word interval, the phone
  /// symbol for a phone interval. It is empty for a silence gap in the
  /// "words" tier (Praat interval tiers must tile the full time range, so
  /// the gaps between words are intervals with empty text).
  final String text;
}

/// One tier of the alignment (phones or words).
class AlignmentTier {
  const AlignmentTier({required this.name, required this.intervals});

  factory AlignmentTier.fromJson(Map<String, dynamic> json) {
    return AlignmentTier(
      name: json['name'] as String,
      intervals: (json['intervals'] as List<dynamic>)
          .map((e) => AlignmentInterval.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }

  @override
  String toString() {
    return 'AlignmentTier(name: $name, intervals: $intervals)';
  }

  Map<String, dynamic> toJson() => {
        'name': name,
        'intervals': intervals.map((e) => e.toJson()).toList(),
      };

  /// Tier name: "phones" or "words"; it is never empty.
  final String name;

  /// The aligned intervals of this tier, in time order.
  final List<AlignmentInterval> intervals;
}

/// The result of [ForcedAligner.compute]: the phone and word boundaries of
/// one utterance, and the same content as a Praat TextGrid.
class ForcedAlignmentResult {
  const ForcedAlignmentResult({
    required this.duration,
    required this.tiers,
    required this.textGrid,
  });

  factory ForcedAlignmentResult.fromJson(Map<String, dynamic> json) {
    return ForcedAlignmentResult(
      duration: (json['duration'] as num).toDouble(),
      tiers: (json['tiers'] as List<dynamic>)
          .map((e) => AlignmentTier.fromJson(e as Map<String, dynamic>))
          .toList(),
      textGrid: json['textGrid'] as String,
    );
  }

  @override
  String toString() {
    return 'ForcedAlignmentResult(duration: $duration, tiers: $tiers, textGrid: $textGrid)';
  }

  Map<String, dynamic> toJson() => {
        'duration': duration,
        'tiers': tiers.map((e) => e.toJson()).toList(),
        'textGrid': textGrid,
      };

  /// Duration of the aligned audio in seconds.
  final double duration;

  /// The alignment tiers: tiers[0].name == "phones", tiers[1].name ==
  /// "words". Empty for an empty result.
  final List<AlignmentTier> tiers;

  /// The alignment as a Praat TextGrid string. Save it to a .TextGrid file
  /// and open it with Praat (https://www.fon.hum.uva.nl/praat/) to
  /// visualize the phone and word boundaries. Empty for an empty result.
  final String textGrid;
}
