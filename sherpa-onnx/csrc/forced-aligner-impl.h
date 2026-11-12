// sherpa-onnx/csrc/forced-aligner-impl.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_FORCED_ALIGNER_IMPL_H_
#define SHERPA_ONNX_CSRC_FORCED_ALIGNER_IMPL_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "kaldi-hmm-gmm/csrc/am-diag-gmm.h"
#include "kaldi-hmm-gmm/csrc/context-dep.h"
#include "kaldi-hmm-gmm/csrc/eigen.h"
#include "kaldi-hmm-gmm/csrc/transition-model.h"
#include "kaldi-native-fbank/csrc/feature-transform.h"
#include "sherpa-onnx/csrc/fa-feature-extractor.h"
#include "sherpa-onnx/csrc/fa-lexicon.h"
#include "sherpa-onnx/csrc/forced-aligner.h"

namespace sherpa_onnx {

// Parsed meta.json of a forced-alignment acoustic model. The members below the
// feature config are the trailing-step and lexicon parameters; see the field
// comments of ForcedAlignerConfig for the file layout.
struct FaModelMeta {
  FaFeatureConfig features;

  // Trailing step of the feature chain. When an LDA
  // transform (lda.mat) is present it takes precedence and the frames are
  // spliced with splice_left/right_context before the transform; otherwise
  // deltas (order 2, window 2) are added when uses_deltas is true.
  bool uses_deltas = false;
  bool uses_splices = false;
  int32_t splice_left_context = 3;
  int32_t splice_right_context = 3;

  // Silence/OOV parameters used when building the lexicon FST.
  float silence_probability = 0.5;
  std::string oov_phone = "spn";
  std::string optional_silence_phone = "sil";
  std::vector<std::string> phones;  // used when phones.txt is missing
};

class ForcedAligner::Impl {
 public:
  explicit Impl(const ForcedAlignerConfig &config);
  ~Impl();

  std::unique_ptr<OfflineStream> CreateStream() const;
  ForcedAlignmentResult Compute(const OfflineStream *stream,
                                const std::string &text) const;

 private:
  // Word/pronunciation resolution of one utterance. The same token can get
  // different pronunciations in different contexts, so graph words are keyed
  // by (token, pronunciations): ids are 1..N over the distinct pairs.
  struct ResolvedWords {
    std::vector<int32_t> word_ids;    // per text token
    std::vector<std::string> labels;  // word label per id (index id-1)
    std::unordered_map<int32_t, std::vector<std::vector<int32_t>>>
        pronunciations;
  };

  std::string ModelFile(const char *name) const;

  void CheckModelFiles() const;

  void LoadLdaTransform();

  void LoadAcousticModel();

  void LoadContextTree();

  void LoadPhoneTable();

  void PrintDebugInfo() const;

  std::vector<int32_t> DisambigSyms() const;

  khg::FloatMatrix ComputeFeatures(const OfflineStream *stream,
                                   bool debug) const;

  ResolvedWords ResolveWords(const std::vector<std::string> &tokens,
                             bool debug) const;

  fst::VectorFst<fst::StdArc> CompileAlignGraph(
      const ResolvedWords &resolved) const;

  std::vector<int32_t> Align(fst::VectorFst<fst::StdArc> *graph,
                             const khg::FloatMatrix &feats,
                             const ForcedAlignmentOptions &opts, bool debug,
                             std::vector<int32_t> *aligned_words) const;

  ForcedAlignmentResult BuildResult(const std::vector<int32_t> &alignment,
                                    const std::vector<int32_t> &aligned_words,
                                    const ResolvedWords &resolved) const;

  ForcedAlignerConfig config_;
  FaModelMeta meta_;
  FaLexicon lexicon_;

  khg::TransitionModel trans_model_;
  khg::AmDiagGmm am_;
  khg::ContextDependency tree_;
  std::unique_ptr<knf::FeatureTransform> lda_;  // one-shot load; may be null
  bool use_lda_ = false;

  // phone symbol <-> model phone id (from phones.txt when present)
  std::unordered_map<std::string, int32_t> phone2id_;
  std::unordered_map<int32_t, std::string> id2phone_;
  int32_t sil_id_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FORCED_ALIGNER_IMPL_H_
