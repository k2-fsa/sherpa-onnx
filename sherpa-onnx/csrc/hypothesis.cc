/**
 * Copyright (c)  2023-2024  Xiaomi Corporation
 * Copyright (c)  2023  Pingfeng Luo
 */

#include "sherpa-onnx/csrc/hypothesis.h"

#include <algorithm>
#include <utility>

namespace sherpa_onnx {

void Hypotheses::Add(Hypothesis hyp, bool use_ctc /*= false */) {
  auto key = hyp.Key();
  auto it = hyps_dict_.find(key);
  if (it == hyps_dict_.end()) {
    hyps_dict_[key] = std::move(hyp);
  } else {
    if (use_ctc) {
      it->second.log_prob_b =
          LogAdd<double>()(it->second.log_prob_b, hyp.log_prob_b);
      it->second.log_prob_nb =
          LogAdd<double>()(it->second.log_prob_nb, hyp.log_prob_nb);
    } else {
      it->second.log_prob = LogAdd<double>()(it->second.log_prob, hyp.log_prob);
    }
  }
}

Hypothesis Hypotheses::GetMostProbable(bool length_norm,
                                       bool use_ctc /*= false */) const {
  if (length_norm == false) {
    return std::max_element(
               hyps_dict_.begin(), hyps_dict_.end(),
               [use_ctc](const auto &left, const auto &right) -> bool {
                 return left.second.TotalLogProb(use_ctc) <
                        right.second.TotalLogProb(use_ctc);
               })
        ->second;
  } else {
    // for length_norm is true
    return std::max_element(
               hyps_dict_.begin(), hyps_dict_.end(),
               [use_ctc](const auto &left, const auto &right) -> bool {
                 return left.second.TotalLogProb(use_ctc) /
                            left.second.ys.size() <
                        right.second.TotalLogProb(use_ctc) /
                            right.second.ys.size();
               })
        ->second;
  }
}

std::vector<Hypothesis> Hypotheses::GetTopK(int32_t k, bool length_norm,
                                            bool use_ctc /*= false*/) const {
  k = std::max(k, 1);
  k = std::min(k, Size());

  std::vector<Hypothesis> all_hyps = Vec();

  if (length_norm == false) {
    std::partial_sort(all_hyps.begin(), all_hyps.begin() + k, all_hyps.end(),
                      [use_ctc](const auto &a, const auto &b) {
                        return a.TotalLogProb(use_ctc) >
                               b.TotalLogProb(use_ctc);
                      });
  } else {
    // for length_norm is true
    std::partial_sort(all_hyps.begin(), all_hyps.begin() + k, all_hyps.end(),
                      [use_ctc](const auto &a, const auto &b) {
                        return a.TotalLogProb(use_ctc) / a.ys.size() >
                               b.TotalLogProb(use_ctc) / b.ys.size();
                      });
  }

  return {all_hyps.begin(), all_hyps.begin() + k};
}

const std::vector<int32_t> GetHypsRowSplits(
    const std::vector<Hypotheses> &hyps) {
  std::vector<int32_t> row_splits;
  row_splits.reserve(hyps.size() + 1);

  row_splits.push_back(0);
  int32_t s = 0;
  for (const auto &h : hyps) {
    s += h.Size();
    row_splits.push_back(s);
  }

  return row_splits;
}

}  // namespace sherpa_onnx
