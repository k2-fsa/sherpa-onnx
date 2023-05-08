// sherpa-onnx/csrc/context-graph.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/context-graph.h"

#include <queue>

namespace sherpa_onnx {
void ContextGraph::Build(const std::vector<std::vector<int32_t>> &token_ids) {
  for (int32_t i = 0; i < token_ids.size(); ++i) {
    auto node = root_;
    for (int32_t j = 0; j < token_ids[i].size(); ++j) {
      int32_t token = token_ids[i][j];
      if (0 == node->next.count(token)) {
        bool is_end = j == token_ids[i].size() - 1;
        node->next[token] = std::make_shared<ContextState>(ContextState(
            token, context_score_, node->total_score + context_score_, is_end));
      }
      node = node->next[token];
    }
  }
  FillFail();
}

std::pair<float, ContextStatePtr> ContextGraph::ForwardOneStep(
    const ContextStatePtr &state, int32_t token) const {
  ContextStatePtr node;
  float score;
  if (1 == state->next.count(token)) {
    node = state->next[token];
    score = node->score;
    if (node->is_end) {
      node = root_;
    }
  } else {
    node = state->fail;
    while (0 == node->next.count(token)) {
      node = node->fail;
      if (-1 == node->token) break;  // root
    }
    if (1 == node->next.count(token)) {
      node = node->next[token];
    }
    score = node->total_score - state->total_score;
    if (node->is_end) {
      node = root_;
    }
  }
  return std::make_pair(score, node);
}

std::pair<float, ContextStatePtr> ContextGraph::Finalize(
    const ContextStatePtr &state) const {
  float score = root_->total_score - state->total_score;
  if (state->is_end) {
    score = 0;
  }
  return std::make_pair(score, root_);
}

void ContextGraph::FillFail() {
  std::queue<ContextStatePtr> node_queue;
  for (auto kv : root_->next) {
    kv.second->fail = root_;
    node_queue.push(kv.second);
  }
  while (!node_queue.empty()) {
    ContextStatePtr current_node = node_queue.front();
    node_queue.pop();
    ContextStatePtr current_fail = current_node->fail;
    for (auto kv : current_node->next) {
      ContextStatePtr fail = current_fail;
      if (1 == current_fail->next.count(kv.first)) {
        fail = current_fail->next[kv.first];
      }
      kv.second->fail = fail;
      node_queue.push(kv.second);
    }
  }
}
}  // namespace sherpa_onnx
