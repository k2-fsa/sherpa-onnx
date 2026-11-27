// sherpa-onnx/csrc/context-graph-test.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/context-graph.h"

#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <string>
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
      {"HEHERSHE", 10}, {"HERSHE", 8},  {"HISHE", 5},
      {"SHED", 6},      {"SHELF", 6},   {"HELL", 2},
      {"HELLO", 7},     {"DHRHISQ", 3}, {"THEN", 2}};
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
      {"HEHERSHE", 22.5}, {"HERSHE", 17.5},  {"HISHE", 10.84},
      {"SHED", 18.34},    {"SHELF", 18.34},  {"HELL", 5},
      {"HELLO", 13},      {"DHRHISQ", 5.84}, {"THEN", 5}};
  TestHelper(queries, 5, false);
}

// In non-strict mode, a phrase that is a prefix of another one must not stop
// the longer one from matching.
TEST(ContextGraph, TestPrefixNonStrict) {
  std::vector<std::string> contexts_str({"CAT", "CATCH"});
  std::vector<std::vector<int32_t>> contexts;
  for (const auto &c : contexts_str) {
    contexts.emplace_back(c.begin(), c.end());
  }
  auto context_graph = ContextGraph(contexts, 1);

  auto score = [&context_graph](const std::string &query, bool strict_mode) {
    float total_scores = 0;
    auto state = context_graph.Root();
    for (auto q : query) {
      auto res = context_graph.ForwardOneStep(state, q, strict_mode);
      total_scores += std::get<0>(res);
      state = std::get<1>(res);
    }
    return total_scores + context_graph.Finalize(state).first;
  };

  // CATCH gets its own bonus (5) on top of CAT's (3). An unfinished CATCH
  // keeps only CAT's bonus, and so does CAT followed by something else.
  auto queries = std::map<std::string, float>{
      {"CAT", 3}, {"CATS", 3}, {"CATC", 3}, {"CATCH", 8}, {"CATCHES", 8}};
  for (const auto &iter : queries) {
    EXPECT_EQ(score(iter.first, false), iter.second) << iter.first;
    EXPECT_EQ(score(iter.first, true), iter.second) << iter.first;
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
