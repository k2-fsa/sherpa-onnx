// sherpa-onnx/csrc/vad-stream-state-test.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/vad-stream-state.h"

#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/vad-model.h"

namespace sherpa_onnx {

namespace {

// A deterministic model so the tests need no VAD model files. The k-th
// window is speech if pattern[k] is true. The window count is part of the
// state, so a stream that skips saving or restoring it gets windows from
// the wrong position of the pattern.
class FakeVadModel : public VadModel {
 public:
  explicit FakeVadModel(std::vector<bool> pattern)
      : pattern_(std::move(pattern)) {
    // One state tensor so the stream round-trips the state like it does
    // with the real models.
    state_.states.push_back({0.0f});
    state_.state_shapes.push_back({1});
  }

  bool IsSpeech(const float* samples, int32_t n) override {
    int32_t k = state_.current_sample;
    state_.current_sample += 1;
    return pattern_[k % pattern_.size()];
  }

  float Compute(const float* samples, int32_t n) override { return 0.5; }

  void Reset() override {
    state_ = {};
    state_.states.push_back({0.0f});
    state_.state_shapes.push_back({1});
  }

  int32_t WindowSize() const override { return window_size_; }

  int32_t WindowShift() const override { return window_shift_; }

  int32_t MinSilenceDurationSamples() const override { return 0; }

  int32_t MinSpeechDurationSamples() const override { return 0; }

  void SetMinSilenceDuration(float s) override { min_silence_ = s; }

  void SetThreshold(float threshold) override { threshold_ = threshold; }

  VadModelState GetState() const override { return state_; }

  void SetState(VadModelState state) override { state_ = std::move(state); }

 private:
  std::vector<bool> pattern_;
  int32_t window_size_ = 512;
  int32_t window_shift_ = 512;
  float threshold_ = 0.5;
  float min_silence_ = 0;
  VadModelState state_;
};

VadModelConfig TestConfig() {
  VadModelConfig config;
  config.sample_rate = 16000;
  // The state only reads the config fields; no model file is loaded.
  config.silero_vad.model = "silero_vad.onnx";
  config.silero_vad.min_silence_duration = 0;
  config.silero_vad.min_speech_duration = 0;
  config.silero_vad.max_speech_duration = 30;
  return config;
}

std::vector<float> Window(float value) {
  return std::vector<float>(512, value);
}

}  // namespace

TEST(VadStreamState, WindowBoundary) {
  VadStreamState state(TestConfig(), 60);
  FakeVadModel model({false});

  auto samples = Window(0);
  samples.resize(511);

  state.AcceptWaveform(16000, samples.data(), samples.size());
  EXPECT_FALSE(state.IsReady(&model));

  // Not ready: Process does nothing.
  state.Process(&model);
  EXPECT_FALSE(state.IsReady(&model));

  samples.push_back(0);
  state.AcceptWaveform(16000, samples.data(), 1);
  EXPECT_TRUE(state.IsReady(&model));

  state.Process(&model);
  EXPECT_FALSE(state.IsReady(&model));
}

TEST(VadStreamState, SavesSegmentWhenSpeechEnds) {
  VadStreamState state(TestConfig(), 60);
  FakeVadModel model({true, true, false});

  for (int32_t i = 0; i < 3; ++i) {
    auto samples = Window(0.01f * (i + 1));
    state.AcceptWaveform(16000, samples.data(), samples.size());
    ASSERT_TRUE(state.IsReady(&model));
    state.Process(&model);
  }

  EXPECT_FALSE(state.IsSpeechDetected());
  EXPECT_FALSE(state.Empty());

  const SpeechSegment& segment = state.Front();
  EXPECT_EQ(segment.start, 0);
  ASSERT_EQ(segment.samples.size(), 3u * 512);
  // The first two windows are speech and the last one is silence, which
  // min_silence_duration of 0 keeps in the segment.
  EXPECT_EQ(segment.samples[0], 0.01f);
  EXPECT_EQ(segment.samples[512], 0.02f);
  EXPECT_EQ(segment.samples[1024], 0.03f);

  state.Pop();
  EXPECT_TRUE(state.Empty());
}

TEST(VadStreamState, TwoStreamsAreIndependent) {
  VadModelConfig config = TestConfig();
  FakeVadModel model({true, false, false, true});

  VadStreamState first(config, 60);
  VadStreamState second(config, 60);

  for (int32_t i = 0; i < 3; ++i) {
    auto samples = Window(0.01f);
    first.AcceptWaveform(16000, samples.data(), samples.size());
    first.Process(&model);
  }
  EXPECT_FALSE(first.Empty());
  EXPECT_EQ(first.Front().samples.size(), 2u * 512);

  // The shared model must resume from the second stream's state, not from
  // the last window the first stream processed.
  auto samples = Window(0.02f);
  second.AcceptWaveform(16000, samples.data(), samples.size());
  second.Process(&model);
  EXPECT_TRUE(second.IsSpeechDetected());

  second.Flush(&model);
  EXPECT_FALSE(second.Empty());
  EXPECT_EQ(second.Front().start, 0);
  EXPECT_EQ(second.Front().samples.size(), 512u);

  // The first stream's queued segment is untouched.
  EXPECT_FALSE(first.Empty());
  EXPECT_EQ(first.Front().samples.size(), 2u * 512);

  // The shared model now sits at the second stream's position. Processing
  // the first stream again produces the right window only if its own saved
  // state is restored into the shared model: position 3 of the pattern is
  // speech, while the position the second stream left is not.
  samples = Window(0.03f);
  first.AcceptWaveform(16000, samples.data(), samples.size());
  first.Process(&model);
  EXPECT_TRUE(first.IsSpeechDetected());
}

TEST(VadStreamState, FlushSavesOpenSegment) {
  VadStreamState state(TestConfig(), 60);
  FakeVadModel model({true, true});

  for (int32_t i = 0; i < 2; ++i) {
    auto samples = Window(0.01f);
    state.AcceptWaveform(16000, samples.data(), samples.size());
    state.Process(&model);
  }
  EXPECT_TRUE(state.IsSpeechDetected());
  EXPECT_TRUE(state.Empty());

  std::vector<float> tail(100, 0.01f);
  state.AcceptWaveform(16000, tail.data(), tail.size());
  EXPECT_FALSE(state.IsReady(&model));

  state.Flush(&model);
  EXPECT_FALSE(state.Empty());
  EXPECT_EQ(state.Front().start, 0);
  EXPECT_EQ(state.Front().samples.size(), 2u * 512 + 100);
}

TEST(VadStreamState, ResetClearsEverything) {
  VadStreamState state(TestConfig(), 60);
  FakeVadModel model({true});

  auto samples = Window(0.01f);
  state.AcceptWaveform(16000, samples.data(), samples.size());
  state.Process(&model);
  EXPECT_TRUE(state.IsSpeechDetected());

  state.Reset(&model);
  EXPECT_TRUE(state.Empty());
  EXPECT_FALSE(state.IsSpeechDetected());
  EXPECT_FALSE(state.IsReady(&model));

  // The stream can be used again.
  state.AcceptWaveform(16000, samples.data(), samples.size());
  EXPECT_TRUE(state.IsReady(&model));
  state.Process(&model);
  EXPECT_TRUE(state.IsSpeechDetected());
}

TEST(VadStreamState, ResamplesInput) {
  VadStreamState state(TestConfig(), 60);
  FakeVadModel model({false});

  // 3:1 resampling: 4710 samples at 48 kHz are 1570 samples at 16 kHz,
  // enough for three windows; the resampler keeps a few samples back
  // until it is flushed.
  std::vector<float> audio(4710);
  for (size_t i = 0; i < audio.size(); ++i) {
    audio[i] = 0.1f * (i % 7);
  }
  state.AcceptWaveform(48000, audio.data(), audio.size());

  for (int32_t i = 0; i < 3; ++i) {
    ASSERT_TRUE(state.IsReady(&model));
    state.Process(&model);
  }
  // Three windows consumed one by one, the resampler has already emitted
  // everything for this input, so the remainder is below a window.
  EXPECT_FALSE(state.IsReady(&model));

  // The samples the resampler held back are buffered for the next
  // windows, not dropped.
  state.Flush(&model);
  EXPECT_TRUE(state.Empty());
}

}  // namespace sherpa_onnx
