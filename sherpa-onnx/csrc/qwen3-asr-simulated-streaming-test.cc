// sherpa-onnx/csrc/qwen3-asr-simulated-streaming-test.cc
//
// Copyright (c) 2026 Qing Peng

#include <algorithm>
#include <filesystem>  // NOLINT(build/c++17): C++17 RAII fixture directory.
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/qwen-asr-tokenizer.h"
#include "sherpa-onnx/csrc/qwen3-asr-continuation.h"
#include "sherpa-onnx/csrc/qwen3-asr-simulated-streaming-impl.h"

namespace sherpa_onnx {
namespace {

// Exact entries from Qwen3-ASR-0.6B's byte vocabulary; no model download.
class PrefixTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::random_device random;
    for (int i = 0; i < 100; ++i) {
      dir_ = std::filesystem::temp_directory_path() /
             ("sherpa-qwen-prefix-" + std::to_string(random()));
      if (std::filesystem::create_directory(dir_)) break;
      dir_.clear();
    }
    ASSERT_FALSE(dir_.empty());
    Write("vocab.json", R"({"A":32,"B":33,"C":34,"D":35,"E":36,
      "F":37,"G":38,"H":39,"I":40,"J":41,"Hello":9707,
      "ä":160,"¸":116,"Ń":255,"ð":172,"Ł":253,"ĺ":246,"Ģ":222,
      "ï":171,"¿":123,"½":121,"Ġ":220,"language":11528,
      "English":22574,"Chinese":44923,"<asr_text>":151704,
      "<|im_end|>":151645})");
    Write("merges.txt", "#version: 0.2\n");
    Write("tokenizer_config.json", "{}");
    tokenizer_ = std::make_unique<QwenAsrTokenizer>(dir_.string());
  }
  void TearDown() override {
    tokenizer_.reset();
    if (!dir_.empty()) std::filesystem::remove_all(dir_);
  }
  void Write(const char *name, const std::string &text) {
    std::ofstream f(dir_ / name);
    f << text;
    ASSERT_TRUE(f.good());
  }
  std::filesystem::path dir_;
  std::shared_ptr<QwenAsrTokenizer> tokenizer_;
};

using Status = Qwen3ASRStreamingStatus;
using Reason = Qwen3ASRStopReason;

TEST_F(PrefixTest, ExactBoundariesAndFinish) {
  for (int n : {0, 1, 31999, 32000, 32001, 100800}) {
    std::vector<int> ends;
    std::vector<float> pcm(n, 0.2f);
    Qwen3ASRStreamingState state(
        {}, tokenizer_, [&](const auto &audio, const auto &request) {
          ends.push_back(audio.size());
          EXPECT_EQ(audio.back(), pcm[audio.size() - 1]);
          Qwen3ASRContinuationResult r;
          r.stop_reason = Reason::kEos;
          r.raw_output_ids = {32, 33, 34, 35, 36, 37, 38};
          r.result.text = "ABCDEFG";
          if (ends.size() <= 2)
            EXPECT_TRUE(request.prefix_ids.empty());
          else
            EXPECT_EQ(request.prefix_ids, (std::vector<int64_t>{32, 33}));
          return r;
        });
    auto out = state.AcceptWaveform(pcm.data(), n);
    EXPECT_EQ(out.status, Status::kOk);
    EXPECT_EQ(out.accepted_samples, n);
    EXPECT_EQ(out.processed_samples, n / 32000 * 32000);
    EXPECT_EQ(ends.size(), n / 32000);
    EXPECT_EQ(out.steps.size(), ends.size());
    for (size_t j = 0; j < ends.size(); ++j)
      EXPECT_EQ(ends[j], (j + 1) * 32000);
    auto final = state.Finish();
    EXPECT_EQ(final.status, Status::kOk);
    ASSERT_EQ(final.updates.size(), 1);
    EXPECT_TRUE(final.updates[0].is_final);
    EXPECT_EQ(final.processed_samples, n);
    EXPECT_EQ(final.updates[0].audio_end_sample, n);
    if (n) EXPECT_EQ(ends.back(), n);
    const auto calls = ends.size();
    EXPECT_TRUE(state.Finish().updates.empty());
    EXPECT_EQ(ends.size(), calls);
    EXPECT_EQ(state.AcceptWaveform(nullptr, 0).status, Status::kInputFinished);
    state.Reset();
    EXPECT_EQ(state.GetResult().revision, 0);
    EXPECT_EQ(state.GetResult().audio_end_sample, 0);
    EXPECT_FALSE(state.GetResult().is_final);
    EXPECT_TRUE(state.GetResult().result.text.empty());
    EXPECT_EQ(state.AcceptWaveform(nullptr, 0).status, Status::kOk);
  }
}

TEST_F(PrefixTest, PacketPartitionDoesNotChangeTrace) {
  using Trace = std::vector<std::pair<int, std::vector<int64_t>>>;
  std::vector<Trace> traces;
  std::vector<float> pcm(100800);
  for (size_t i = 0; i < pcm.size(); ++i)
    pcm[i] = static_cast<float>(i) / pcm.size();
  for (int mode = 0; mode < 3; ++mode) {
    Trace trace;
    Qwen3ASRStreamingState state(
        {}, tokenizer_, [&](const auto &a, const auto &r) {
          trace.emplace_back(a.size(), r.prefix_ids);
          EXPECT_TRUE(std::equal(a.begin(), a.end(), pcm.begin()));
          Qwen3ASRContinuationResult result;
          result.stop_reason = Reason::kEos;
          result.raw_output_ids = {32, 33, 34, 35, 36, 37, 38, 39};
          return result;
        });
    std::mt19937 random(42);
    size_t offset = 0;
    while (offset < pcm.size()) {
      size_t n =
          mode == 0 ? 320 : (mode == 1 ? 1 + random() % 64000 : pcm.size());
      n = std::min(n, pcm.size() - offset);
      EXPECT_EQ(state.AcceptWaveform(pcm.data() + offset, n).status,
                Status::kOk);
      offset += n;
    }
    state.Finish();
    traces.push_back(trace);
  }
  EXPECT_EQ(traces[0], traces[1]);
  EXPECT_EQ(traces[0], traces[2]);
  EXPECT_EQ(traces[0], (Trace{{32000, {}},
                              {64000, {}},
                              {96000, {32, 33, 34}},
                              {100800, {32, 33, 34}}}));
}

TEST_F(PrefixTest, InvalidPacketsAreAtomicAndRecoverable) {
  int calls = 0;
  Qwen3ASRStreamingState s({}, tokenizer_, [&](const auto &, const auto &) {
    ++calls;
    Qwen3ASRContinuationResult r;
    r.stop_reason = Reason::kEos;
    return r;
  });
  for (float v : {std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::infinity(), -1.01f, 1.01f}) {
    std::vector<float> a(32000, 0);
    a.back() = v;
    auto r = s.AcceptWaveform(a.data(), a.size());
    EXPECT_EQ(r.status, Status::kInvalidArgument);
    EXPECT_EQ(r.accepted_samples, 0);
  }
  EXPECT_EQ(s.AcceptWaveform(nullptr, 1).status, Status::kInvalidArgument);
  EXPECT_EQ(s.AcceptWaveform(nullptr, -1).status, Status::kInvalidArgument);
  std::vector<float> big(480001, 0);
  EXPECT_EQ(s.AcceptWaveform(big.data(), big.size()).status,
            Status::kAudioLimit);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(s.AcceptWaveform(big.data(), 32000).updates.size(), 1);
}

TEST_F(PrefixTest, FailureKeepsSuccessfulUpdatesAndCannotBecomeFinal) {
  for (auto reason : {Reason::kContextLimit, Reason::kTokenLimit,
                      Reason::kRepetitionStopped, Reason::kDecodeError}) {
    int calls = 0;
    Qwen3ASRStreamingState s({}, tokenizer_, [&](const auto &, const auto &) {
      Qwen3ASRContinuationResult r;
      r.stop_reason = ++calls == 1 ? Reason::kEos : reason;
      r.result.text = calls == 1 ? "good" : "failed";
      return r;
    });
    std::vector<float> a(100800, 0);
    auto r = s.AcceptWaveform(a.data(), a.size());
    EXPECT_NE(r.status, Status::kOk);
    ASSERT_EQ(r.updates.size(), 1);
    EXPECT_EQ(r.accepted_samples, 100800);
    EXPECT_EQ(r.processed_samples, 32000);
    ASSERT_EQ(r.steps.size(), 2);
    EXPECT_EQ(r.steps.back().audio_end_sample, 64000);
    EXPECT_EQ(s.GetResult().result.text, "good");
    EXPECT_EQ(s.Finish().status, r.status);
    EXPECT_TRUE(s.Finish().updates.empty());
    EXPECT_EQ(s.AcceptWaveform(a.data(), 1).status, r.status);
    EXPECT_FALSE(s.GetResult().is_final);
    EXPECT_EQ(calls, 2);
  }
}

TEST_F(PrefixTest, ExceptionsBecomeDecodeFailureAndStreamsStayIndependent) {
  Qwen3ASRStreamingState failed(
      {}, tokenizer_,
      [](const auto &, const auto &) -> Qwen3ASRContinuationResult {
        throw std::runtime_error("decode");
      });
  Qwen3ASRStreamingState good({}, tokenizer_, [](const auto &, const auto &) {
    Qwen3ASRContinuationResult r;
    r.stop_reason = Reason::kEos;
    r.result.text = "ok";
    return r;
  });
  std::vector<float> a(32000, 0);
  auto failure = failed.AcceptWaveform(a.data(), a.size());
  EXPECT_EQ(failure.status, Status::kDecodeError);
  ASSERT_EQ(failure.steps.size(), 1);
  EXPECT_EQ(failure.steps[0].audio_end_sample, 32000);
  EXPECT_EQ(failure.steps[0].stop_reason, "decode_error");
  EXPECT_GT(failure.steps[0].service_seconds, 0);
  EXPECT_EQ(good.AcceptWaveform(a.data(), 1).processed_samples, 0);
  auto final = good.Finish();
  ASSERT_EQ(final.updates.size(), 1);
  EXPECT_EQ(final.updates[0].result.text, "ok");
  EXPECT_TRUE(failed.GetResult().result.text.empty());

  int calls = 0;
  Qwen3ASRStreamingState later(
      {}, tokenizer_,
      [&](const auto &, const auto &) -> Qwen3ASRContinuationResult {
        if (++calls == 2) throw std::runtime_error("decode");
        Qwen3ASRContinuationResult r;
        r.stop_reason = Reason::kEos;
        r.result.text = "good";
        return r;
      });
  a.resize(64000);
  auto out = later.AcceptWaveform(a.data(), a.size());
  EXPECT_EQ(out.status, Status::kDecodeError);
  ASSERT_EQ(out.updates.size(), 1);
  ASSERT_EQ(out.steps.size(), 2);
  EXPECT_EQ(out.steps[1].audio_end_sample, 64000);
  EXPECT_EQ(out.steps[1].stop_reason, "decode_error");
  EXPECT_EQ(out.processed_samples, 32000);
}

TEST(StreamingConfiguration, RejectsBeforeLoadingModel) {
  OfflineRecognizerConfig m;
  for (int invalid : {0, -1, 480001}) {
    Qwen3ASRSimulatedStreamingConfig c;
    c.max_utterance_samples = invalid;
    EXPECT_THROW(Qwen3ASRSimulatedStreamingRecognizer(m, c),
                 std::invalid_argument);
  }
  EXPECT_THROW((void)Qwen3ASRSimulatedStreamingRecognizer(m),
               std::invalid_argument);
  m.model_config.qwen3_asr.conv_frontend = "unloaded";
  m.model_config.qwen3_asr.encoder = "unloaded";
  m.model_config.qwen3_asr.decoder = "unloaded";
  m.model_config.qwen3_asr.tokenizer = "unloaded";
  for (int field = 0; field < 4; ++field) {
    Qwen3ASRSimulatedStreamingConfig c;
    if (field == 0) c.chunk_size_samples = 0;
    if (field == 1) c.unfixed_chunk_num = -1;
    if (field == 2) c.unfixed_token_num = -1;
    if (field == 3) c.chunk_size_samples = 480001;
    EXPECT_THROW(Qwen3ASRSimulatedStreamingRecognizer(m, c),
                 std::invalid_argument);
  }
  m.model_config.provider = "cuda";
  EXPECT_THROW((void)Qwen3ASRSimulatedStreamingRecognizer(m),
               std::invalid_argument);
  m.model_config.provider = "cpu";
  m.model_config.qwen3_asr.temperature = 0.5;
  EXPECT_THROW((void)Qwen3ASRSimulatedStreamingRecognizer(m),
               std::invalid_argument);
}

TEST_F(PrefixTest, WarmupUsesEmptyPrefix) {
  for (int j : {0, 1}) {
    EXPECT_TRUE(
        MakeQwen3ASRPrefix({32, 33, 34, 35, 36, 37, 38}, j, 2, 5, *tokenizer_)
            .empty());
  }
  EXPECT_EQ(
      MakeQwen3ASRPrefix({32, 33, 34, 35, 36, 37, 38}, 2, 2, 5, *tokenizer_),
      (std::vector<int64_t>{32, 33}));
}

TEST_F(PrefixTest, InvalidLanguageBytesRemainSerializable) {
  bool invalid = false;
  auto r = FormatQwen3ASRContinuation({11528, 220, 255, 151704, 32}, "",
                                      *tokenizer_, &invalid);
  EXPECT_TRUE(invalid);
  EXPECT_EQ(r.text, "A");
  EXPECT_EQ(r.lang, "\xef\xbf\xbd");
  EXPECT_NO_THROW((void)nlohmann::json::parse(r.AsJsonString()));
}

TEST_F(PrefixTest, RollbackUsesIdsNotCharacters) {
  EXPECT_EQ(
      MakeQwen3ASRPrefix({9707, 32, 33, 34, 35, 36}, 2, 2, 5, *tokenizer_),
      (std::vector<int64_t>{9707}));
}

TEST_F(PrefixTest, RollbackZero) {
  EXPECT_EQ(MakeQwen3ASRPrefix({32, 33}, 2, 2, 0, *tokenizer_),
            (std::vector<int64_t>{32, 33}));
}

TEST_F(PrefixTest, RollbackBeyondLength) {
  EXPECT_TRUE(MakeQwen3ASRPrefix({32}, 2, 2, 5, *tokenizer_).empty());
  EXPECT_TRUE(MakeQwen3ASRPrefix({}, 2, 2, 0, *tokenizer_).empty());
}

TEST_F(PrefixTest, RollbackRetreatsToUtf8Boundary) {
  // 中: E4 B8 AD, split across three byte tokens. Emoji: F0 9F 98 80.
  ASSERT_EQ(tokenizer_->Decode({160, 116, 255}), "中");
  EXPECT_EQ(MakeQwen3ASRPrefix({32, 160, 116, 255}, 2, 2, 1, *tokenizer_),
            (std::vector<int64_t>{32}));
  EXPECT_EQ(MakeQwen3ASRPrefix({32, 172, 253, 246, 222}, 2, 2, 1, *tokenizer_),
            (std::vector<int64_t>{32}));
  // A stray continuation byte in the interior is invalid even with no pending
  // bytes at the end. Strict validation must retreat past it.
  EXPECT_EQ(MakeQwen3ASRPrefix({32, 255, 33}, 2, 2, 0, *tokenizer_),
            (std::vector<int64_t>{32}));
}

TEST_F(PrefixTest, ValidReplacementCharacterIsNotRejected) {
  EXPECT_EQ(MakeQwen3ASRPrefix({171, 123, 121, 32}, 2, 2, 1, *tokenizer_),
            (std::vector<int64_t>{171, 123, 121}));
}

TEST_F(PrefixTest, InvalidParametersAreRejected) {
  EXPECT_THROW(MakeQwen3ASRPrefix({}, -1, 2, 5, *tokenizer_),
               std::invalid_argument);
  EXPECT_THROW(MakeQwen3ASRPrefix({}, 0, -1, 5, *tokenizer_),
               std::invalid_argument);
  EXPECT_THROW(MakeQwen3ASRPrefix({}, 0, 2, -1, *tokenizer_),
               std::invalid_argument);
}

TEST_F(PrefixTest, DisplayCombinesRetainedIdsAndSuffixOnce) {
  std::vector<int64_t> raw{11528, 220, 44923, 151704, 160, 116, 255, 32};
  bool invalid = false;
  auto r = FormatQwen3ASRContinuation(raw, "", *tokenizer_, &invalid);
  EXPECT_EQ(r.text, "中A");
  EXPECT_EQ(r.lang, "Chinese");
  EXPECT_FALSE(invalid);
  // Display changes (including homophone replacement) cannot change raw IDs.
  r.text = "仲A";
  EXPECT_EQ(MakeQwen3ASRPrefix(raw, 2, 2, 1, *tokenizer_),
            (std::vector<int64_t>{11528, 220, 44923, 151704, 160, 116, 255}));
  EXPECT_EQ(
      FormatQwen3ASRContinuation({9707, 32}, "English", *tokenizer_, &invalid)
          .text,
      "HelloA");
}

TEST_F(PrefixTest, IncompleteAutomaticScaffoldIsHidden) {
  bool invalid = false;
  for (const auto &ids : std::vector<std::vector<int64_t>>{
           {11528}, {11528, 220}, {11528, 220, 22574}}) {
    EXPECT_TRUE(FormatQwen3ASRContinuation(ids, "", *tokenizer_, &invalid)
                    .text.empty());
  }
}

TEST_F(PrefixTest, InvalidBytesAreReplacedAndFlagged) {
  bool invalid = false;
  auto r = FormatQwen3ASRContinuation({32, 255, 33, 160}, "Chinese",
                                      *tokenizer_, &invalid);
  EXPECT_EQ(r.text,
            "A\xef\xbf\xbd"
            "B\xef\xbf\xbd");
  EXPECT_TRUE(invalid);
  r = FormatQwen3ASRContinuation({171, 123, 121}, "", *tokenizer_, &invalid);
  EXPECT_EQ(r.text, "\xef\xbf\xbd");
  EXPECT_FALSE(invalid);
}

}  // namespace
}  // namespace sherpa_onnx
