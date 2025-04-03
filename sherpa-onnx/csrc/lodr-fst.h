// sherpa-onnx/csrc/lodr-fst.h
//
// Contains code copied from icefall/utils/ngram_lm.py
// Copyright (c)  2023 Xiaomi Corporation


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

class LODRFST {
 public:
  explicit LODRFST(const std::string &fst_path);

  std::pair<std::vector<int>, std::vector<float>> get_next_states_costs(
    int state, int label);

  void ComputeScore(float scale, Hypothesis *hyp, int32_t offset);

 private:
  fst::StdVectorFst YsToFst(const std::vector<int64_t> &ys, int32_t offset);

  std::vector<std::tuple<int, float>> process_backoff_arcs(
    int state, float cost);

  std::optional<std::tuple<int, float>> next_states_costs_no_backoff(
    int state, int label);


  int backoff_id;
  std::unique_ptr<fst::StdConstFst> fst_;
};

class LODRStateCost {
 public:
  explicit LODRStateCost(
    LODRFST* fst,
    std::unordered_map<int, float> state_cost = {});

  LODRStateCost forward_one_step(int label);

  float lm_score() const;

 private:
  LODRFST* fst_;
  std::unordered_map<int, float> state_cost_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_LODR_FST_H_
