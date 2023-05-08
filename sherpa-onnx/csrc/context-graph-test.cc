// sherpa-onnx/csrc/context-graph-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/context-graph.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(ContextGraph, TestBasic) {
  std::vector<std::string> contexts_str({"HE", "SHE", "HIS", "HERS"});
  std::vector<std::vector<int32_t>> contexts;
  for (int32_t i = 0; i < contexts_str.size(); ++i) {
    std::vector<int32_t> tmp;
    for (int32_t j = 0; j < contexts_str[i].size(); ++j) {
      tmp.push_back((int32_t)contexts_str[i][j]);
    }
    contexts.push_back(tmp);
  }
  auto context_graph = ContextGraph(2);
  context_graph.Build(contexts);
  auto res = context_graph.ForwardOneStep(context_graph.Root(), (int32_t)'H');
  EXPECT_EQ(res.first, 2);
  EXPECT_EQ(res.second->token, (int32_t)'H');

  res = context_graph.ForwardOneStep(res.second, (int32_t)'I');
  EXPECT_EQ(res.first, 2);
  EXPECT_EQ(res.second->token, (int32_t)'I');

  res = context_graph.ForwardOneStep(res.second, (int32_t)'S');
  EXPECT_EQ(res.first, 2);
  EXPECT_EQ(res.second->token, -1);

  res = context_graph.ForwardOneStep(context_graph.Root(), (int32_t)'S');
  EXPECT_EQ(res.first, 2);
  EXPECT_EQ(res.second->token, (int32_t)'S');

  res = context_graph.ForwardOneStep(res.second, (int32_t)'H');
  EXPECT_EQ(res.first, 2);
  EXPECT_EQ(res.second->token, (int32_t)'H');

  res = context_graph.ForwardOneStep(res.second, (int32_t)'D');
  EXPECT_EQ(res.first, -4);
  EXPECT_EQ(res.second->token, -1);

  res = context_graph.ForwardOneStep(context_graph.Root(), (int32_t)'D');
  EXPECT_EQ(res.first, 0);
  EXPECT_EQ(res.second->token, -1);
}

}  // namespace sherpa_onnx
