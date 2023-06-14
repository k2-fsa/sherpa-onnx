// sherpa-onnx/csrc/context-graph.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_
#define SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sherpa_onnx {

class ContextState;
class ContextGraph;
using ContextStatePtr = std::shared_ptr<ContextState>;
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

struct ContextState {
  int32_t token;
  float token_score;
  float node_score;
  float local_node_score;
  bool is_end;
  std::unordered_map<int32_t, ContextStatePtr> next;
  std::weak_ptr<ContextState> fail;
  std::weak_ptr<ContextState> output;

  ContextState() = default;
  ContextState(int32_t token, float token_score, float node_score,
               float local_node_score, bool is_end)
      : token(token),
        token_score(token_score),
        node_score(node_score),
        local_node_score(local_node_score),
        is_end(is_end) {}
};

class ContextGraph {
 public:
  ContextGraph() = default;
  explicit ContextGraph(float context_score) : context_score_(context_score) {
    is_populated_ = false;
    root_ = std::make_shared<ContextState>(-1, 0, 0, 0, false);
    root_->fail = root_;
  }

  void Build(const std::vector<std::vector<int32_t>> &token_ids);

  std::pair<float, ContextStatePtr> ForwardOneStep(const ContextStatePtr &state,
                                                   int32_t token_id) const;
  std::pair<float, ContextStatePtr> Finalize(
      const ContextStatePtr &state) const;

  ContextStatePtr Root() const { return root_; }

 private:
  float context_score_;
  ContextStatePtr root_;
  bool is_populated_;
  void FillFailOutput() const;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_
