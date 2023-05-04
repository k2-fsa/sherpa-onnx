/**
 * Copyright (c)  2023  Xiaomi Corporation
 * Copyright (c)  2023  Pingfeng Luo
 *
 */

#ifndef SHERPA_ONNX_CSRC_HYPOTHESIS_H_
#define SHERPA_ONNX_CSRC_HYPOTHESIS_H_

#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

struct Hypothesis {
  // The predicted tokens so far. Newly predicated tokens are appended.
  std::vector<int64_t> ys;

  // timestamps[i] contains the frame number after subsampling
  // on which ys[i] is decoded.
  std::vector<int32_t> timestamps;

  // The total score of ys in log space.
  // It contains only acoustic scores
  double log_prob = 0;

  // LM log prob if any.
  double lm_log_prob = 0;

  int32_t cur_scored_pos = 0;      // cur scored tokens by RNN LM
  CopyableOrtValue rnnlm_state_h;  // states for RNN LM
  CopyableOrtValue rnnlm_state_c;  // states for RNN LM
  // the minimum of tokens in a chunk for streaming RNN LM
  int32_t lm_rescore_min_chunk = 2;
  bool lm_states_inited = false;

  int32_t num_trailing_blanks = 0;

  Hypothesis() = default;
  Hypothesis(const std::vector<int64_t> &ys, double log_prob)
      : ys(ys), log_prob(log_prob) {}

  double TotalLogProb() const { return log_prob + lm_log_prob; }

  // If two Hypotheses have the same `Key`, then they contain
  // the same token sequence.
  std::string Key() const {
    // TODO(fangjun): Use a hash function?
    std::ostringstream os;
    std::string sep = "-";
    for (auto i : ys) {
      os << i << sep;
      sep = "-";
    }
    return os.str();
  }

  // For debugging
  std::string ToString() const {
    std::ostringstream os;
    os << "(" << Key() << ", " << log_prob << ")";
    return os.str();
  }
};

class Hypotheses {
 public:
  Hypotheses() = default;

  explicit Hypotheses(std::vector<Hypothesis> hyps) {
    for (auto &h : hyps) {
      hyps_dict_[h.Key()] = std::move(h);
    }
  }

  explicit Hypotheses(std::unordered_map<std::string, Hypothesis> hyps_dict)
      : hyps_dict_(std::move(hyps_dict)) {}

  // Add hyp to this object. If it already exists, its log_prob
  // is updated with the given hyp using log-sum-exp.
  void Add(Hypothesis hyp);

  // Get the hyp that has the largest log_prob.
  // If length_norm is true, hyp's log_prob is divided by
  // len(hyp.ys) before comparison.
  Hypothesis GetMostProbable(bool length_norm) const;

  // Get the k hyps that have the largest log_prob.
  // If length_norm is true, hyp's log_prob is divided by
  // len(hyp.ys) before comparison.
  std::vector<Hypothesis> GetTopK(int32_t k, bool length_norm) const;

  int32_t Size() const { return hyps_dict_.size(); }

  std::string ToString() const {
    std::ostringstream os;
    for (const auto &p : hyps_dict_) {
      os << p.second.ToString() << "\n";
    }
    return os.str();
  }

  const auto begin() const { return hyps_dict_.begin(); }
  const auto end() const { return hyps_dict_.end(); }

  auto begin() { return hyps_dict_.begin(); }
  auto end() { return hyps_dict_.end(); }

  void Clear() { hyps_dict_.clear(); }

 private:
  // Return a list of hyps contained in this object.
  std::vector<Hypothesis> Vec() const {
    std::vector<Hypothesis> ans;
    ans.reserve(hyps_dict_.size());
    for (const auto &p : hyps_dict_) {
      ans.push_back(p.second);
    }
    return ans;
  }

 private:
  using Map = std ::unordered_map<std::string, Hypothesis>;
  Map hyps_dict_;
};

const std::vector<int32_t> GetHypsRowSplits(
    const std::vector<Hypotheses> &hyps);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_HYPOTHESIS_H_
