// sherpa-onnx/csrc/forced-aligner.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FORCED_ALIGNER_H_
#define SHERPA_ONNX_CSRC_FORCED_ALIGNER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct AlignmentInterval {
  float start = 0.0f;
  float end = 0.0f;
  std::string text;
};

struct AlignmentTier {
  std::string name;
  std::vector<AlignmentInterval> intervals;
};

struct ForcedAlignmentResult {
  float duration = 0.0f;

  // tiers[0].name == "phones", tiers[1].name == "words"
  std::vector<AlignmentTier> tiers;

  // Praat TextGrid
  std::string ToTextGrid() const;
};

// Per-utterance / per-stream alignment knobs, following kalpy's GmmAligner
// defaults. A stream can override any of them via SetOption() before
// Compute(), e.g. SetOption("beam", "20").
struct ForcedAlignmentOptions {
  float beam = 10.0f;
  float retry_beam = 40.0f;
  float transition_scale = 1.0f;
  float self_loop_scale = 0.1f;
  float acoustic_scale = 0.1f;

  void Register(ParseOptions *po);
  std::string ToString() const;
};

struct ForcedAlignerConfig {
  /// Directory of one forced-alignment acoustic model. Required files:
  ///   meta.json    [required] feature chain + silence/OOV parameters
  ///                (parsed once at construction, field by field)
  ///   final.alimdl [required] alignment model (kaldi format:
  ///                TransitionModel + AmDiagGmm; note: NOT final.mdl)
  ///   tree         [required] context dependency tree (ContextDependency)
  ///   lda.mat      [required iff the feature chain is splice+LDA]
  ///   phones.txt   [optional] phone table (falls back to the phones list
  ///                in meta.json when missing)
  ///   (final.mdl, phone_lm.fst, graphemes.txt, rules.yaml etc. are unused;
  ///   missing required files are reported by name)
  std::string model_dir;

  /// Pronunciation dictionary, one pronunciation per line:
  ///   "key [probability columns ...] phone1 phone2 ..."
  /// Probability columns (floating point numbers between the key and the
  /// phones, e.g. MFA .dict files) are ignored. Several lines under one key
  /// = several pronunciations of the same word/character, and ALL are kept:
  /// parsing groups lines by key into a pronunciation list (identical phone
  /// sequences deduplicated) and each pronunciation becomes a parallel
  /// path in L.fst (acoustic selection). This is how Chinese polyphones
  /// avoid text-level guessing.
  /// NOTE: word intervals are recovered along the pronunciation path
  /// actually taken by the alignment; word boundaries are NOT counted with
  /// the phone count of the first pronunciation (pronunciations differ in
  /// length).
  /// Key shape depends on the language: English = words, Chinese = hanzi
  /// characters/phrases (hanzi-keyed, e.g.
  /// scripts/mfa/zh/lars76/*/lexicon.txt), other languages = their writing
  /// units.
  /// L.fst is not shipped with the model; it is built here in kaldi
  /// dictionary form (optional silence + word output symbols).
  std::string lexicon;

  /// Default alignment options; a stream can override them via SetOption().
  ForcedAlignmentOptions options;

  /// true to print debug information: model info while loading, per-utterance
  /// alignment info in Compute(). It can also be enabled for a single stream
  /// via SetOption("debug", "1").
  bool debug = false;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class ForcedAligner {
 public:
  explicit ForcedAligner(const ForcedAlignerConfig &config);
  ~ForcedAligner();

  // Alignment options can be overridden per stream via SetOption(); when
  // unset, the defaults of ForcedAlignerConfig::options are used.
  std::unique_ptr<OfflineStream> CreateStream() const;

  // text is the full utterance transcript. Hanzi text is tokenized per
  // character (multi-character dictionary keys give pronunciation context);
  // anything else is split on whitespace. Alignment options are read from
  // the stream via GetOptionFloat().
  // The stream is passed by pointer (house style).
  // Returns an empty result (no tiers) with a warning when the stream is
  // null, the text is empty, the stream has no features, or word boundaries
  // cannot be recovered from the alignment.
  ForcedAlignmentResult Compute(const OfflineStream *stream,
                                const std::string &text) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FORCED_ALIGNER_H_
