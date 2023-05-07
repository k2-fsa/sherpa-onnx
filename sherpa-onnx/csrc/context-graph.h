// sherpa-onnx/csrc/context-graph.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_
#define SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_

#include <memory>
#include <unordered_map>
#include <vector>

namespace sherpa_onnx {

class ContextState;
using ContextStatePtr = std::shared_ptr<ContextState>;

struct ContextState {
  int32_t token;
  float score;
  float total_score;
  bool is_end;
  std::unordered_map<int32_t, ContextStatePtr> next;
  ContextStatePtr fail;

  ContextState() = default;
  ContextState(int32_t token, float score, float total_score, bool is_end)
      : token(token), score(score), total_score(total_score), is_end(is_end) {}
};

class ContextGraph {
 public:
  ContextGraph() = default;
  explicit ContextGraph(float context_score) : context_score_(context_score) {
    root_ = std::make_shared<ContextState>(ContextState(-1, 0, 0, false));
    root_->fail = root_;
  }
  ~ContextGraph() {}
  void BuildContextGraph(const std::vector<std::vector<int32_t>> &token_ids);

  std::pair<float, ContextStatePtr> ForwardOneStep(const ContextStatePtr &state,
                                                   int32_t token_id) const;
  std::pair<float, ContextStatePtr> Finalize(
      const ContextStatePtr &state) const;

  ContextStatePtr Root() const { return root_; }

 private:
  float context_score_;
  ContextStatePtr root_;
  void FillFail();
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_CONTEXT_GRAPH_H_
