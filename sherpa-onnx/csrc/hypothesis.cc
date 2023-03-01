/**
 * Copyright (c)  2023  Xiaomi Corporation
 *
 */

#include "sherpa-onnx/csrc/hypothesis.h"

#include <algorithm>
#include <utility>

namespace sherpa_onnx {

void Hypotheses::Add(Hypothesis hyp) {
  auto key = hyp.Key();
  auto it = hyps_dict_.find(key);
  if (it == hyps_dict_.end()) {
    hyps_dict_[key] = std::move(hyp);
  } else {
    it->second.log_prob = LogAdd<double>()(it->second.log_prob, hyp.log_prob);
  }
}

Hypothesis Hypotheses::GetMostProbable(bool length_norm) const {
  if (length_norm == false) {
    return std::max_element(hyps_dict_.begin(), hyps_dict_.end(),
                            [](const auto &left, auto &right) -> bool {
                              return left.second.log_prob <
                                     right.second.log_prob;
                            })
        ->second;
  } else {
    // for length_norm is true
    return std::max_element(
               hyps_dict_.begin(), hyps_dict_.end(),
               [](const auto &left, const auto &right) -> bool {
                 return left.second.log_prob / left.second.ys.size() <
                        right.second.log_prob / right.second.ys.size();
               })
        ->second;
  }
}

std::vector<Hypothesis> Hypotheses::GetTopK(int32_t k, bool length_norm) const {
  k = std::max(k, 1);
  k = std::min(k, Size());

  std::vector<Hypothesis> all_hyps = Vec();

  if (length_norm == false) {
    std::partial_sort(
        all_hyps.begin(), all_hyps.begin() + k, all_hyps.end(),
        [](const auto &a, const auto &b) { return a.log_prob > b.log_prob; });
  } else {
    // for length_norm is true
    std::partial_sort(all_hyps.begin(), all_hyps.begin() + k, all_hyps.end(),
                      [](const auto &a, const auto &b) {
                        return a.log_prob / a.ys.size() >
                               b.log_prob / b.ys.size();
                      });
  }

  return {all_hyps.begin(), all_hyps.begin() + k};
}

}  // namespace sherpa_onnx
