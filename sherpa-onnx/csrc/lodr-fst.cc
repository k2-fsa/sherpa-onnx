// sherpa-onnx/csrc/lodr-fst.cc
//
// Contains code copied from icefall/utils/ngram_lm.py
// Copyright (c)  2023 Xiaomi Corporation

#include <algorithm>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/lodr-fst.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/hypothesis.h"

namespace sherpa_onnx {

LODRFST::LODRFST(const std::string &fst_path) {
  fst_ = std::unique_ptr<fst::StdConstFst>(
    CastOrConvertToConstFst(fst::StdVectorFst::Read(fst_path)));
}

std::vector<std::tuple<int, float>> LODRFST::process_backoff_arcs(
    int state, float cost) {
  std::vector<std::tuple<int, float>> ans;
  auto next = next_states_costs_no_backoff(state, backoff_id);
  if (!next.has_value()) {
    return ans;
  }
  auto [next_state, next_cost] = next.value();
  ans.emplace_back(next_state, next_cost + cost);
  auto recursive_result = process_backoff_arcs(next_state, next_cost + cost);
  ans.insert(ans.end(), recursive_result.begin(), recursive_result.end());
  return ans;
}

std::optional<std::tuple<int, float>> LODRFST::next_states_costs_no_backoff(
    int state, int label) {
  fst::ArcIterator<fst::StdConstFst> arc_iter(*fst_, state);
  int num_arcs = fst_->NumArcs(state);

  int left = 0, right = num_arcs - 1;
  while (left <= right) {
    int mid = (left + right) / 2;
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

std::pair<std::vector<int>, std::vector<float>> LODRFST::get_next_states_costs(
    int state, int label) {
  std::vector<int> states = {state};
  std::vector<float> costs = {0};

  auto extra_states_costs = process_backoff_arcs(state, 0);
  for (const auto& [s, c] : extra_states_costs) {
    states.push_back(s);
    costs.push_back(c);
  }

  std::vector<int> next_states;
  std::vector<float> next_costs;
  for (size_t i = 0; i < states.size(); ++i) {
    auto next = next_states_costs_no_backoff(states[i], label);
    if (next.has_value()) {
      auto [ns, nc] = next.value();
      next_states.push_back(ns);
      next_costs.push_back(costs[i] + nc);
    }
  }

  return std::make_pair(next_states, next_costs);
}

void LODRFST::ComputeScore(float scale, Hypothesis *hyp, int32_t offset) {
  if (scale == 0) {
    return;
  }

  using Weight = typename fst::StdArc::Weight;
  using Arc = fst::StdArc;

  // Step 1: Convert the input text into an FST
  fst::StdVectorFst ys = YsToFst(hyp->ys, offset);

  // Step 2: Compose the input text with the rule FST
  fst::StdVectorFst composed_fst;
  fst::Compose(ys, *fst_, &composed_fst);

  // Step 3: Get the best path from the composed FST
  fst::StdVectorFst score_fst;
  fst::ShortestPath(composed_fst, &score_fst, 1);

  // Step 4: Compute the total weight (i.e. score of the hypothesis)
  Weight total_weight = Weight::One();
  auto s = score_fst.Start();
  if (s == fst::kNoStateId) {
    // this is an empty FST
    SHERPA_ONNX_LOG("Empty FST after scoring hyp with LODR! Mismatched FST?");
    return;
  }
  while (score_fst.Final(s) == Weight::Zero()) {
    fst::ArcIterator<fst::Fst<Arc>> aiter(score_fst, s);
    if (aiter.Done()) {
      // not reached final.
      SHERPA_ONNX_LOG("LODR FST failed to reach final state. Mismatched FST?");
      return;
    }

    const auto &arc = aiter.Value();

    total_weight = Times(total_weight, arc.weight);

    s = arc.nextstate;
    if (s == fst::kNoStateId) {
      SHERPA_ONNX_LOG("Transition to invalid state in LODR FST");
      return;
    }

    aiter.Next();
    if (!aiter.Done()) {
      // not a linear FST
      SHERPA_ONNX_LOG("Error in applying LODR FST. Not a linear FST?");
      return;
    }
  }

  if (score_fst.Final(s) != Weight::Zero()) {
    total_weight = Times(total_weight, score_fst.Final(s));
  }

  // Update the hyp score
  hyp->log_prob += scale * total_weight.Value();
}

fst::StdVectorFst LODRFST::YsToFst(
    const std::vector<int64_t> &ys, int32_t offset) {
  using Weight = typename fst::StdArc::Weight;
  using Arc = fst::StdArc;

  fst::StdVectorFst ans;
  ans.ReserveStates(ys.size());

  auto s = ans.AddState();
  ans.SetStart(s);
  for (size_t i = offset; i < ys.size(); ++i) {
    const auto nextstate = ans.AddState();
    ans.AddArc(s, Arc(ys[i], ys[i], Weight::One(), nextstate));
    s = nextstate;
  }

  ans.SetFinal(s, Weight::One());

  return ans;
}

LODRStateCost::LODRStateCost(
    LODRFST* fst, std::unordered_map<int, float> state_cost)
    : fst_(fst) {
  if (state_cost.empty()) {
    state_cost_[0] = 0.0;
  } else {
    state_cost_ = state_cost;
  }
}

LODRStateCost LODRStateCost::forward_one_step(int label) {
  std::unordered_map<int, float> state_cost;
  for (const auto& [s, c] : state_cost_) {
    auto [next_states, next_costs] = fst_->get_next_states_costs(s, label);
    for (size_t i = 0; i < next_states.size(); ++i) {
      int ns = next_states[i];
      float nc = next_costs[i];
      if (state_cost.find(ns) == state_cost.end()) {
        state_cost[ns] = std::numeric_limits<float>::infinity();
      }
      state_cost[ns] = std::min(state_cost[ns], c + nc);
    }
  }
  return LODRStateCost(fst_, state_cost);
}

float LODRStateCost::lm_score() const {
  if (state_cost_.empty()) {
    return -std::numeric_limits<float>::infinity();
  }
  auto min_cost = std::min_element(state_cost_.begin(), state_cost_.end(),
                                   [](const auto& a, const auto& b) {
                                     return a.second < b.second;
                                   });
  return -min_cost->second;
}

}  // namespace sherpa_onnx
