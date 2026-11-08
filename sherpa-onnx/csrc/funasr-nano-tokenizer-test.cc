// sherpa-onnx/csrc/funasr-nano-tokenizer-test.cc
//
// Copyright (c)  2026  sherpa-onnx contributors

#include "sherpa-onnx/csrc/funasr-nano-tokenizer.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

namespace fs = std::filesystem;

// Writes a tiny byte-level BPE tokenizer: single letters a-z, plus merges
// for "ab" and "abc". ASCII letters map to themselves in ByteLevel encoding.
class TinyTokenizerDir {
 public:
  TinyTokenizerDir() {
    dir_ = fs::temp_directory_path() /
           ("funasr-nano-tokenizer-test-" +
            std::to_string(std::random_device()()));
    fs::create_directories(dir_);

    std::string vocab = "{";
    for (char c = 'a'; c <= 'z'; ++c) {
      vocab += "\"" + std::string(1, c) + "\": " + std::to_string(c - 'a') +
               ", ";
    }
    vocab += "\"ab\": 26, \"abc\": 27}";
    Write("vocab.json", vocab);
    Write("merges.txt", "#version: 0.2\na b\nab c\n");
    Write("tokenizer.json", "{\"added_tokens\": []}");
  }

  ~TinyTokenizerDir() {
    std::error_code ec;
    fs::remove_all(dir_, ec);
  }

  std::string Path() const { return dir_.string(); }

 private:
  void Write(const std::string &name, const std::string &content) {
    std::ofstream(dir_ / name, std::ios::binary) << content;
  }

  fs::path dir_;
};

// A distinct lowercase word for every index, so each Encode() call misses
// the BPE cache and inserts into it.
std::string Word(int32_t i) {
  std::string w = "abc";
  do {
    w.push_back(static_cast<char>('a' + i % 26));
    i /= 26;
  } while (i > 0);
  return w;
}

TEST(FunASRNanoTokenizer, AppliesMerges) {
  TinyTokenizerDir dir;
  const FunASRNanoTokenizer tokenizer(dir.Path());
  // "abc" merges fully; "d" stays a single letter.
  EXPECT_EQ(tokenizer.Encode("abcd"), (std::vector<int64_t>{27, 3}));
}

TEST(FunASRNanoTokenizer, ConcurrentEncodeIsSafeAndDeterministic) {
  TinyTokenizerDir dir;
  constexpr int32_t kThreads = 8;
  constexpr int32_t kWordsPerThread = 2000;

  // Expected results from a separate, single-threaded tokenizer.
  const FunASRNanoTokenizer reference(dir.Path());
  std::vector<std::vector<int64_t>> expected(kThreads * kWordsPerThread);
  for (int32_t i = 0; i < kThreads * kWordsPerThread; ++i) {
    expected[i] = reference.Encode(Word(i));
  }

  // One shared tokenizer, as used by a recognizer or TTS instance that is
  // called from several threads.
  const FunASRNanoTokenizer shared(dir.Path());
  std::vector<std::vector<int64_t>> actual(kThreads * kWordsPerThread);
  std::vector<std::thread> threads;
  for (int32_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int32_t k = 0; k < kWordsPerThread; ++k) {
        // Interleave indices so threads insert into the cache concurrently.
        const int32_t i = k * kThreads + t;
        actual[i] = shared.Encode(Word(i));
      }
    });
  }
  for (auto &th : threads) th.join();

  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace sherpa_onnx
