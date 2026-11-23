// sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl-test.cc
//
// Copyright (c)  2026  fra-shipper

#include "sherpa-onnx/csrc/offline-recognizer-qwen3-asr-impl.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/qwen3-asr-continuation.h"
#include "sherpa-onnx/csrc/qwen3-asr-simulated-streaming.h"
#include "sherpa-onnx/csrc/wave-reader.h"

namespace sherpa_onnx {

class QwenContinuationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *dir = std::getenv("QWEN3_TEST_MODEL_DIR");
    if (!dir) GTEST_SKIP() << "Run scripts/qwen3-asr/test-continuation.py";
    dir_ = dir;
    config_.model_config.qwen3_asr.conv_frontend = dir_ + "/conv.onnx";
    config_.model_config.qwen3_asr.encoder = dir_ + "/encoder.onnx";
    config_.model_config.qwen3_asr.decoder = dir_ + "/eos.onnx";
    config_.model_config.qwen3_asr.tokenizer = dir_ + "/tokenizer";
    config_.model_config.num_threads = 1;
  }
  Qwen3ASRContinuationResult Run(OfflineRecognizerQwen3ASRImpl &impl,
                                 const std::vector<int64_t> &prefix,
                                 const std::string &language = "",
                                 int32_t max_len = 512, int32_t max_new = 128) {
    auto s = impl.CreateStream();
    std::vector<float> pcm(32000, 0.1f);
    s->AcceptWaveform(16000, pcm.data(), pcm.size());
    s->SetOption("language", language);
    s->SetOption("max_total_len", std::to_string(max_len));
    s->SetOption("max_new_tokens", std::to_string(max_new));
    return impl.DecodeWithPrefix(s.get(), {prefix, true});
  }
  std::string dir_;
  OfflineRecognizerConfig config_;
};

TEST_F(QwenContinuationTest, NonemptyPrefixAllowsImmediateEos) {
  OfflineRecognizerQwen3ASRImpl impl(config_);
  auto r = Run(impl, {32, 33});
  EXPECT_EQ(r.stop_reason, Qwen3ASRStopReason::kEos);
  EXPECT_EQ(r.generated_tokens, 0);
  EXPECT_EQ(r.decoder_calls, 1);
  EXPECT_EQ(r.raw_output_ids, (std::vector<int64_t>{32, 33}));
  EXPECT_EQ(r.result.text, "AB");
  auto legacy = impl.CreateStream();
  std::vector<float> pcm(32000, 0.1f);
  legacy->AcceptWaveform(16000, pcm.data(), pcm.size());
  auto *p = legacy.get();
  impl.DecodeStreams(&p, 1);
  EXPECT_EQ(legacy->GetResult().text, "A");
  EXPECT_EQ(Run(impl, {}).result.text, "A");
}

TEST_F(QwenContinuationTest, ExactCapacityKeepsAudioAndPrefix) {
  OfflineRecognizerQwen3ASRImpl impl(config_);
  for (const auto &language : {std::string(), std::string("Chinese")}) {
    auto r = Run(impl, {32, 33}, language);
    ASSERT_GT(r.prompt_tokens, 2);
    EXPECT_EQ(r.prompt_tokens, Run(impl, {}, language).prompt_tokens + 2);
    EXPECT_EQ(Run(impl, {32, 33}, language, r.prompt_tokens + 1).stop_reason,
              Qwen3ASRStopReason::kEos);
    auto fail = Run(impl, {32, 33}, language, r.prompt_tokens);
    EXPECT_EQ(fail.stop_reason, Qwen3ASRStopReason::kContextLimit);
    EXPECT_EQ(fail.prompt_tokens, r.prompt_tokens);
    EXPECT_EQ(fail.generated_tokens, 0);
  }
}

TEST_F(QwenContinuationTest, FullPromptPositionsFreshKvAndTwoNewTokens) {
  config_.model_config.qwen3_asr.decoder = dir_ + "/observe.onnx";
  OfflineRecognizerQwen3ASRImpl impl(config_);
  for (int i = 0; i < 2; ++i) {
    auto r = Run(impl, {32, 33});
    EXPECT_EQ(r.stop_reason, Qwen3ASRStopReason::kEos);
    EXPECT_EQ(r.generated_tokens, 2);
    EXPECT_EQ(r.decoder_calls, 3);
    EXPECT_GT(r.encoder_seconds, 0);
    EXPECT_GT(r.decoder_seconds, 0);
    EXPECT_EQ(r.raw_output_ids, (std::vector<int64_t>{32, 33, 34, 35}));
    EXPECT_EQ(r.result.text, "ABCD");
  }
  EXPECT_EQ(Run(impl, {32, 33}, "", 512, 1).stop_reason,
            Qwen3ASRStopReason::kTokenLimit);
  auto n = Run(impl, {32, 33}).prompt_tokens;
  EXPECT_EQ(Run(impl, {32, 33}, "", n + 1).stop_reason,
            Qwen3ASRStopReason::kContextLimit);
  auto full = Run(impl, {32, 33}, "", n + 1, 2);
  EXPECT_EQ(full.stop_reason, Qwen3ASRStopReason::kContextLimit);
  EXPECT_EQ(full.generated_tokens, 1);
  EXPECT_EQ(full.raw_output_ids, (std::vector<int64_t>{32, 33, 34}));
  // With one available position a new D followed by EOS is still valid:
  // the final cache slot can be used to confirm EOS, but not append more text.
  auto one = Run(impl, {34});
  auto eos = Run(impl, {34}, "", one.prompt_tokens + 1, 2);
  EXPECT_EQ(eos.stop_reason, Qwen3ASRStopReason::kEos);
  EXPECT_EQ(eos.generated_tokens, 1);
  EXPECT_EQ(eos.result.text, "CD");
}

TEST_F(QwenContinuationTest, RepetitionAcrossPrefixBoundary) {
  config_.model_config.qwen3_asr.decoder = dir_ + "/repeat.onnx";
  OfflineRecognizerQwen3ASRImpl impl(config_);
  auto r = Run(impl, std::vector<int64_t>(63, 32));
  EXPECT_EQ(r.stop_reason, Qwen3ASRStopReason::kRepetitionStopped);
  EXPECT_EQ(r.generated_tokens, 1);
  EXPECT_EQ(r.raw_output_ids.size(), 64);
  config_.model_config.qwen3_asr.decoder = dir_ + "/eos.onnx";
  OfflineRecognizerQwen3ASRImpl eos(config_);
  EXPECT_EQ(Run(eos, std::vector<int64_t>(64, 32)).stop_reason,
            Qwen3ASRStopReason::kEos);
}

TEST_F(QwenContinuationTest, SilentAndShortInputClearPrefix) {
  config_.model_config.qwen3_asr.conv_frontend = dir_ + "/silence.onnx";
  OfflineRecognizerQwen3ASRImpl impl(config_);
  auto r = Run(impl, {32, 33});
  EXPECT_EQ(r.stop_reason, Qwen3ASRStopReason::kNoSpeech);
  EXPECT_TRUE(r.raw_output_ids.empty());
  EXPECT_TRUE(r.result.text.empty());
  auto s = impl.CreateStream();
  float sample = 0;
  s->AcceptWaveform(16000, &sample, 1);
  r = impl.DecodeWithPrefix(s.get(), {{32, 33}, true});
  EXPECT_EQ(r.stop_reason, Qwen3ASRStopReason::kInsufficientFrames);
  EXPECT_TRUE(r.raw_output_ids.empty());
}

TEST_F(QwenContinuationTest, ExplicitEmptyHotwordsDisablesInheritance) {
  config_.model_config.qwen3_asr.hotwords = "Artificial,Intelligence";
  OfflineRecognizerQwen3ASRImpl impl(config_);
  auto inherited = Run(impl, {32});
  auto s = impl.CreateStream();
  std::vector<float> pcm(32000, 0.1f);
  s->AcceptWaveform(16000, pcm.data(), pcm.size());
  s->SetOption("hotwords", "");
  auto r = impl.DecodeWithPrefix(s.get(), {{32}, true});
  EXPECT_LT(r.prompt_tokens, inherited.prompt_tokens);
  EXPECT_EQ(r.result.text, "A");
}

TEST_F(QwenContinuationTest, StrictDigitalSilenceSkipsBiasedEncoder) {
  // The fixture encoder deliberately emits nonzero features even for silence.
  // Silence must be recognized in the audio frontend, not inferred from a
  // trained encoder's biased output.
  OfflineRecognizerQwen3ASRImpl impl(config_);
  auto s = impl.CreateStream();
  std::vector<float> pcm(32000, 0);
  s->AcceptWaveform(16000, pcm.data(), pcm.size());
  auto r = impl.DecodeWithPrefix(s.get(), {{32, 33}, true});
  EXPECT_EQ(r.stop_reason, Qwen3ASRStopReason::kNoSpeech);
  EXPECT_TRUE(r.raw_output_ids.empty());
  EXPECT_TRUE(r.result.text.empty());
  EXPECT_EQ(r.decoder_calls, 0);
}

TEST_F(QwenContinuationTest, MalformedFrontendIsAFailure) {
  for (const char *name : {"malformed", "empty"}) {
    config_.model_config.qwen3_asr.conv_frontend = dir_ + "/" + name + ".onnx";
    OfflineRecognizerQwen3ASRImpl impl(config_);
    EXPECT_EQ(Run(impl, {32, 33}).stop_reason,
              Qwen3ASRStopReason::kDecodeError);
    Qwen3ASRSimulatedStreamingRecognizer recognizer(config_);
    auto stream = recognizer.CreateStream();
    std::vector<float> pcm(32000, 0.1f);
    auto out = stream->AcceptWaveform(pcm.data(), pcm.size());
    EXPECT_EQ(out.status, Qwen3ASRStreamingStatus::kDecodeError);
    EXPECT_TRUE(out.updates.empty());
    EXPECT_EQ(out.processed_samples, 0);
    EXPECT_TRUE(stream->Finish().updates.empty());
  }
}

TEST_F(QwenContinuationTest, StreamOutlivesRecognizerAndOptionsAreImmutable) {
  auto recognizer =
      std::make_unique<Qwen3ASRSimulatedStreamingRecognizer>(config_);
  Qwen3ASRSimulatedStreamingOptions opt;
  opt.language = "English";
  auto a = recognizer->CreateStream(opt);
  opt.language = "Chinese";
  auto b = recognizer->CreateStream(opt);
  opt.language = "invalid";
  EXPECT_THROW(recognizer->CreateStream(opt), std::invalid_argument);
  opt.language = "";
  opt.hotwords = std::string(16385, 'x');
  EXPECT_THROW(recognizer->CreateStream(opt), std::invalid_argument);
  opt.hotwords = "\xff";
  EXPECT_THROW(recognizer->CreateStream(opt), std::invalid_argument);
  recognizer.reset();
  std::vector<float> pcm(32000, 0.1f);
  EXPECT_EQ(a->AcceptWaveform(pcm.data(), pcm.size()).status,
            Qwen3ASRStreamingStatus::kOk);
  EXPECT_EQ(b->AcceptWaveform(pcm.data(), 17).status,
            Qwen3ASRStreamingStatus::kOk);
  auto ra = a->Finish();
  auto rb = b->Finish();
  ASSERT_EQ(ra.updates.size(), 1);
  ASSERT_EQ(rb.updates.size(), 1);
  EXPECT_EQ(ra.updates[0].result.lang, "English");
  // Too-short audio deliberately clears the result and language.
  EXPECT_TRUE(rb.updates[0].result.text.empty());
  a->Reset();
  EXPECT_TRUE(a->GetResult().result.text.empty());
}

TEST(QwenRealContinuation, ChineseAndEnglishAllLogits) {
  const char *root = std::getenv("QWEN3_TEST_RESOURCE_ROOT");
  const char *zh = std::getenv("QWEN3_TEST_ZH_WAV");
  const char *en = std::getenv("QWEN3_TEST_EN_WAV");
  if (!root || !zh || !en) GTEST_SKIP() << "Optional real-model integration";
  std::string dir(root);
  const bool int8 = std::getenv("QWEN3_TEST_INT8") != nullptr;
  std::string suffix = int8 ? ".int8.onnx" : ".onnx";
  OfflineRecognizerConfig c;
  auto &q = c.model_config.qwen3_asr;
  q.conv_frontend = dir + "/exports/all/conv_frontend.onnx";
  q.encoder = dir + "/exports/all/encoder" + suffix;
  q.decoder = dir + "/exports/all/decoder" + suffix;
  q.tokenizer = dir + "/checkpoint/5eb144179a02acc5e5ba31e748d22b0cf3e303b0";
  c.model_config.num_threads = 4;
  OfflineRecognizerQwen3ASRImpl impl(c);
  for (const char *wav : {zh, en}) {
    int32_t rate = 0;
    bool ok = false;
    auto pcm = ReadWave(wav, &rate, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(rate, 16000);
    ASSERT_GT(pcm.size(), 96000);
    auto early = impl.CreateStream();
    early->AcceptWaveform(16000, pcm.data(), 64000);
    auto partial = impl.DecodeWithPrefix(early.get(), {{}, true});
    ASSERT_EQ(partial.stop_reason, Qwen3ASRStopReason::kEos);
    auto prefix =
        MakeQwen3ASRPrefix(partial.raw_output_ids, 2, 2, 5, impl.Tokenizer());
    ASSERT_FALSE(prefix.empty());
    auto s = impl.CreateStream();
    s->AcceptWaveform(16000, pcm.data(), pcm.size());
    auto r = impl.DecodeWithPrefix(s.get(), {prefix, true});
    EXPECT_EQ(r.stop_reason, Qwen3ASRStopReason::kEos);
    EXPECT_GE(r.generated_tokens, 2);
    ASSERT_GE(r.raw_output_ids.size(), prefix.size());
    EXPECT_TRUE(
        std::equal(prefix.begin(), prefix.end(), r.raw_output_ids.begin()));
    EXPECT_FALSE(r.result.text.empty());
    auto empty = impl.CreateStream();
    empty->AcceptWaveform(16000, pcm.data(), pcm.size());
    auto strict = impl.DecodeWithPrefix(empty.get(), {{}, true});
    auto legacy = impl.CreateStream();
    legacy->AcceptWaveform(16000, pcm.data(), pcm.size());
    auto *ptr = legacy.get();
    impl.DecodeStreams(&ptr, 1);
    EXPECT_EQ(strict.result.text, legacy->GetResult().text);
    EXPECT_EQ(strict.result.tokens, legacy->GetResult().tokens);
  }
}

// Regression test for https://github.com/k2-fsa/sherpa-onnx/issues/3509
//
// When every frame of audio_features is silence, TrimAudioFeatures() must
// report that via |all_silent| so that GenerateText() can short-circuit to
// an empty result before any hotwords/language prompt tokens are built.
// Previously the all-silent case was indistinguishable from "nothing needed
// trimming", so decoding proceeded and the hotwords/language prompt could
// bias the LLM decoder into hallucinating text for silent audio.
TEST(TrimAudioFeatures, AllSilentSetsFlag) {
  Ort::AllocatorWithDefaultOptions allocator;

  constexpr int32_t kFrames = 5;
  constexpr int32_t kDim = 4;
  std::array<int64_t, 3> shape{1, kFrames, kDim};
  Ort::Value audio_features =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *p = audio_features.GetTensorMutableData<float>();
  std::memset(p, 0, sizeof(float) * kFrames * kDim);

  bool all_silent = false;
  Ort::Value trimmed =
      TrimAudioFeatures(std::move(audio_features), allocator, &all_silent);

  EXPECT_TRUE(all_silent);

  auto trimmed_shape = trimmed.GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(trimmed_shape.size(), 3u);
  EXPECT_EQ(trimmed_shape[1], kFrames);
}

TEST(TrimAudioFeatures, TrailingSilenceIsTrimmedAndFlagStaysFalse) {
  Ort::AllocatorWithDefaultOptions allocator;

  constexpr int32_t kFrames = 5;
  constexpr int32_t kValidFrames = 3;
  constexpr int32_t kDim = 4;
  std::array<int64_t, 3> shape{1, kFrames, kDim};
  Ort::Value audio_features =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *p = audio_features.GetTensorMutableData<float>();
  std::memset(p, 0, sizeof(float) * kFrames * kDim);
  for (int32_t a = 0; a < kValidFrames; ++a) {
    p[a * kDim] = 1.0f;
  }

  bool all_silent = false;
  Ort::Value trimmed =
      TrimAudioFeatures(std::move(audio_features), allocator, &all_silent);

  EXPECT_FALSE(all_silent);

  auto trimmed_shape = trimmed.GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(trimmed_shape.size(), 3u);
  EXPECT_EQ(trimmed_shape[1], kValidFrames);
}

TEST(TrimAudioFeatures, NoTrailingSilenceFlagStaysFalse) {
  Ort::AllocatorWithDefaultOptions allocator;

  constexpr int32_t kFrames = 3;
  constexpr int32_t kDim = 4;
  std::array<int64_t, 3> shape{1, kFrames, kDim};
  Ort::Value audio_features =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *p = audio_features.GetTensorMutableData<float>();
  for (int32_t i = 0; i < kFrames * kDim; ++i) {
    p[i] = 1.0f;
  }

  bool all_silent = false;
  Ort::Value trimmed =
      TrimAudioFeatures(std::move(audio_features), allocator, &all_silent);

  EXPECT_FALSE(all_silent);

  auto trimmed_shape = trimmed.GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(trimmed_shape.size(), 3u);
  EXPECT_EQ(trimmed_shape[1], kFrames);
}

}  // namespace sherpa_onnx
