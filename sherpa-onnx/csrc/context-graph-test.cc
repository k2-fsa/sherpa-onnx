// sherpa-onnx/csrc/context-graph-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/context-graph.h"

#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

static void TestHelper(const std::map<std::string, float> &queries, float score,
                       bool strict_mode) {
  std::vector<std::string> contexts_str(
      {"S", "HE", "SHE", "SHELL", "HIS", "HERS", "HELLO", "THIS", "THEM"});
  std::vector<std::vector<int32_t>> contexts;
  std::vector<float> scores;
  for (int32_t i = 0; i < contexts_str.size(); ++i) {
    contexts.emplace_back(contexts_str[i].begin(), contexts_str[i].end());
    scores.push_back(std::round(score / contexts_str[i].size() * 100) / 100);
  }
  auto context_graph = ContextGraph(contexts, 1, scores);

  for (const auto &iter : queries) {
    float total_scores = 0;
    auto state = context_graph.Root();
    for (auto q : iter.first) {
      auto res = context_graph.ForwardOneStep(state, q, strict_mode);
      total_scores += std::get<0>(res);
      state = std::get<1>(res);
    }
    auto res = context_graph.Finalize(state);
    EXPECT_EQ(res.second->token, -1);
    total_scores += res.first;
    EXPECT_EQ(total_scores, iter.second);
  }
}

TEST(ContextGraph, TestBasic) {
  auto queries = std::map<std::string, float>{
      {"HEHERSHE", 14}, {"HERSHE", 12}, {"HISHE", 9},
      {"SHED", 6},      {"SHELF", 6},   {"HELL", 2},
      {"HELLO", 7},     {"DHRHISQ", 4}, {"THEN", 2}};
  TestHelper(queries, 0, true);
}

TEST(ContextGraph, TestBasicNonStrict) {
  auto queries = std::map<std::string, float>{
      {"HEHERSHE", 7}, {"HERSHE", 5}, {"HISHE", 5},   {"SHED", 3}, {"SHELF", 3},
      {"HELL", 2},     {"HELLO", 2},  {"DHRHISQ", 3}, {"THEN", 2}};
  TestHelper(queries, 0, false);
}

TEST(ContextGraph, TestCustomize) {
  auto queries = std::map<std::string, float>{
      {"HEHERSHE", 35.84}, {"HERSHE", 30.84},  {"HISHE", 24.18},
      {"SHED", 18.34},     {"SHELF", 18.34},   {"HELL", 5},
      {"HELLO", 13},       {"DHRHISQ", 10.84}, {"THEN", 5}};
  TestHelper(queries, 5, true);
}

TEST(ContextGraph, TestCustomizeNonStrict) {
  auto queries = std::map<std::string, float>{
      {"HEHERSHE", 20}, {"HERSHE", 15},    {"HISHE", 10.84},
      {"SHED", 10},     {"SHELF", 10},     {"HELL", 5},
      {"HELLO", 5},     {"DHRHISQ", 5.84}, {"THEN", 5}};
  TestHelper(queries, 5, false);
}

// A decoder that emits several tokens on one frame (TDT tokens with duration
// 0) must not let a phrase earn its bonus again and again without consuming
// frames. In a query, '|' starts a new frame.
TEST(ContextGraph, TestOneBonusPerFrame) {
  std::vector<std::string> contexts_str({"ABCD"});
  std::vector<std::vector<int32_t>> contexts;
  for (const auto &c : contexts_str) {
    contexts.emplace_back(c.begin(), c.end());
  }
  auto context_graph = ContextGraph(contexts, 1);

  auto score = [&context_graph](const std::string &query, bool per_frame) {
    float total_scores = 0;
    float withheld = 0;
    bool first_on_frame = true;
    auto state = context_graph.Root();
    for (auto q : query) {
      if (q == '|') {
        first_on_frame = true;
        continue;
      }
      auto res = per_frame ? context_graph.ForwardOneStepOnFrame(
                                 state, q, first_on_frame, &withheld, false)
                           : context_graph.ForwardOneStep(state, q, false);
      total_scores += std::get<0>(res);
      state = std::get<1>(res);
      first_on_frame = false;
    }
    return total_scores + context_graph.Finalize(state).first + withheld;
  };

  // {query, score with ForwardOneStep, score with ForwardOneStepOnFrame}
  std::vector<std::tuple<std::string, float, float>> queries = {
      // One token per frame: the same as before.
      {"A|B|C|D", 4, 4},
      {"A|B|C|D|A|B|C|D", 8, 8},
      // A phrase repeated on one frame earns its bonus per token, without
      // consuming frames; now only the first token of the frame gets one.
      {"ABCD", 4, 1},
      {"ABCDABCDABCD", 12, 1},
      {"ABCD|ABCD", 8, 2},
      // An unfinished match still adds up to 0.
      {"ABC", 0, 0},
      {"ABCX", 0, 0},
      {"A|BC", 0, 0},
  };
  for (const auto &q : queries) {
    EXPECT_EQ(score(std::get<0>(q), false), std::get<1>(q)) << std::get<0>(q);
    EXPECT_EQ(score(std::get<0>(q), true), std::get<2>(q)) << std::get<0>(q);
  }
}

TEST(ContextGraph, Benchmark) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int32_t> char_dist(0, 25);
  std::uniform_int_distribution<int32_t> len_dist(3, 8);
  for (int32_t num = 10; num <= 10000; num *= 10) {
    std::vector<std::vector<int32_t>> contexts;
    for (int32_t i = 0; i < num; ++i) {
      std::vector<int32_t> tmp;
      int32_t word_len = len_dist(mt);
      for (int32_t j = 0; j < word_len; ++j) {
        tmp.push_back(char_dist(mt));
      }
      contexts.push_back(std::move(tmp));
    }
    auto start = std::chrono::high_resolution_clock::now();
    auto context_graph = ContextGraph(contexts, 1);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    SHERPA_ONNX_LOGE("Construct context graph for %d item takes %d us.", num,
                     static_cast<int32_t>(duration.count()));
  }
}

}  // namespace sherpa_onnx
