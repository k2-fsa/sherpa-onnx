// sherpa-onnx/csrc/lodr-fst.cc
//
// Contains code copied from icefall/utils/ngram_lm.py
// Copyright (c)  2023 Xiaomi Corporation
//
// Copyright (c)  2025 Tilde SIA (Askars Salimbajevs)

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/lodr-fst.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/hypothesis.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

int32_t LodrFst::FindBackoffId() {
  // assume that the backoff id is the only input label with epsilon output

  for (int32_t state = 0; state < fst_->NumStates(); ++state) {
    fst::ArcIterator<fst::StdConstFst> arc_iter(*fst_, state);
    for ( ; !arc_iter.Done(); arc_iter.Next()) {
      const auto& arc = arc_iter.Value();
      if (arc.olabel == 0) {  // Check if the output label is epsilon (0)
        return arc.ilabel;    // Return the input label
      }
    }
  }

  return -1;  // Return -1 if no such input symbol is found
}

LodrFst::LodrFst(const std::string &fst_path, int32_t backoff_id)
    : backoff_id_(backoff_id) {
  fst_ = std::unique_ptr<fst::StdConstFst>(
    CastOrConvertToConstFst(fst::StdVectorFst::Read(fst_path)));

  if (backoff_id < 0) {
    // backoff_id_ is not provided, find it automatically
    backoff_id_ = FindBackoffId();
    if (backoff_id_ < 0) {
      std::string err_msg = "Failed to initialize LODR: No backoff arc found";
      SHERPA_ONNX_LOGE("%s", err_msg.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }
}

std::vector<std::tuple<int32_t, float>> LodrFst::ProcessBackoffArcs(
  int32_t state, float cost) {
  std::vector<std::tuple<int32_t, float>> ans;
  auto next = GetNextStatesCostsNoBackoff(state, backoff_id_);
  if (!next.has_value()) {
    return ans;
  }
  auto [next_state, next_cost] = next.value();
  ans.emplace_back(next_state, next_cost + cost);
  auto recursive_result = ProcessBackoffArcs(next_state, next_cost + cost);
  ans.insert(ans.end(), recursive_result.begin(), recursive_result.end());
  return ans;
}

std::optional<std::tuple<int32_t, float>> LodrFst::GetNextStatesCostsNoBackoff(
  int32_t state, int32_t label) {
  fst::ArcIterator<fst::StdConstFst> arc_iter(*fst_, state);
  int32_t num_arcs = fst_->NumArcs(state);

  int32_t left = 0, right = num_arcs - 1;
  while (left <= right) {
    int32_t mid = (left + right) / 2;
    arc_iter.Seek(mid);
    auto arc = arc_iter.Value();
    if (arc.ilabel < label) {
      left = mid + 1;
    } else if (arc.ilabel > label) {
      right = mid - 1;
    } else {
      return std::make_tuple(arc.nextstate, arc.weight.Value());
    }
  }
  return std::nullopt;
}

std::pair<std::vector<int32_t>, std::vector<float>> LodrFst::GetNextStateCosts(
  int32_t state, int32_t label) {
  std::vector<int32_t> states = {state};
  std::vector<float> costs = {0};

  auto extra_states_costs = ProcessBackoffArcs(state, 0);
  for (const auto& [s, c] : extra_states_costs) {
    states.push_back(s);
    costs.push_back(c);
  }

  std::vector<int32_t> next_states;
  std::vector<float> next_costs;
  for (size_t i = 0; i < states.size(); ++i) {
    auto next = GetNextStatesCostsNoBackoff(states[i], label);
    if (next.has_value()) {
      auto [ns, nc] = next.value();
      next_states.push_back(ns);
      next_costs.push_back(costs[i] + nc);
    }
  }

  return std::make_pair(next_states, next_costs);
}

void LodrFst::ComputeScore(float scale, Hypothesis *hyp, int32_t offset) {
  if (scale == 0) {
    return;
  }

  hyp->lodr_state = std::make_unique<LodrStateCost>(this);

  // Walk through the FST with the input text from the hypothesis
  for (size_t i = offset; i < hyp->ys.size(); ++i) {
    *hyp->lodr_state = hyp->lodr_state->ForwardOneStep(hyp->ys[i]);
  }

  float lodr_score = hyp->lodr_state->FinalScore();

  if (lodr_score == -std::numeric_limits<float>::infinity()) {
    SHERPA_ONNX_LOGE("Failed to compute LODR. Empty or mismatched FST?");
    return;
  }

  // Update the hyp score
  hyp->log_prob += scale * lodr_score;
}

float LodrFst::GetFinalCost(int32_t state) {
  auto final_weight = fst_->Final(state);
  if (final_weight == fst::StdArc::Weight::Zero()) {
    return 0.0;
  }
  return final_weight.Value();
}

LodrStateCost::LodrStateCost(
    LodrFst* fst, const std::unordered_map<int32_t, float> &state_cost)
    : fst_(fst) {
  if (state_cost.empty()) {
    state_cost_[0] = 0.0;
  } else {
    state_cost_ = state_cost;
  }
}

LodrStateCost LodrStateCost::ForwardOneStep(int32_t label) {
  std::unordered_map<int32_t, float> state_cost;
  for (const auto& [s, c] : state_cost_) {
    auto [next_states, next_costs] = fst_->GetNextStateCosts(s, label);
    for (size_t i = 0; i < next_states.size(); ++i) {
      int32_t ns = next_states[i];
      float nc = next_costs[i];
      if (state_cost.find(ns) == state_cost.end()) {
        state_cost[ns] = std::numeric_limits<float>::infinity();
      }
      state_cost[ns] = std::min(state_cost[ns], c + nc);
    }
  }
  return LodrStateCost(fst_, state_cost);
}

float LodrStateCost::Score() const {
  if (state_cost_.empty()) {
    return -std::numeric_limits<float>::infinity();
  }
  auto min_cost = std::min_element(state_cost_.begin(), state_cost_.end(),
                                   [](const auto& a, const auto& b) {
                                     return a.second < b.second;
                                   });
  return -min_cost->second;
}

float LodrStateCost::FinalScore() const {
  if (state_cost_.empty()) {
    return -std::numeric_limits<float>::infinity();
  }
  auto min_cost = std::min_element(state_cost_.begin(), state_cost_.end(),
                                   [](const auto& a, const auto& b) {
                                     return a.second < b.second;
                                   });
  return -(min_cost->second +
           fst_->GetFinalCost(min_cost->first));
}

}  // namespace sherpa_onnx
