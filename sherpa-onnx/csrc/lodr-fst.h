// sherpa-onnx/csrc/lodr-fst.h
//
// Contains code copied from icefall/utils/ngram_lm.py
// Copyright (c)  2023 Xiaomi Corporation
//
// Copyright (c)  2025 Tilde SIA (Askars Salimbajevs)


#ifndef SHERPA_ONNX_CSRC_LODR_FST_H_
#define SHERPA_ONNX_CSRC_LODR_FST_H_

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <limits>
#include <algorithm>
#include <utility>

#include "kaldifst/csrc/kaldi-fst-io.h"

namespace sherpa_onnx {

class Hypothesis;

class LodrFst {
 public:
  explicit LodrFst(const std::string &fst_path);

  std::pair<std::vector<int32_t>, std::vector<float>> GetNextStateCosts(
    int state, int label);

  void ComputeScore(float scale, Hypothesis *hyp, int32_t offset);

 private:
  fst::StdVectorFst YsToFst(const std::vector<int64_t> &ys, int32_t offset);

  std::vector<std::tuple<int, float>> ProcessBackoffArcs(
    int state, float cost);

  std::optional<std::tuple<int32_t, float>> GetNextStatesCostsNoBackoff(
    int state, int label);


  int32_t backoff_id_ = 0;
  std::unique_ptr<fst::StdConstFst> fst_;
};

class LodrStateCost {
 public:
  explicit LodrStateCost(
    LodrFst* fst,
    std::unordered_map<int, float> state_cost = {});

    LodrStateCost ForwardOneStep(int label);

  float Score() const;

 private:
  LodrFst* fst_;
  std::unordered_map<int, float> state_cost_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_LODR_FST_H_
