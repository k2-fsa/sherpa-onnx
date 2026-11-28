// sherpa-onnx/csrc/online-silero-vad.h
//
// Copyright (c)  2026  Manickavela Arumugam
//
// Silero VAD for the online recognizer: one shared ONNX session per
// recognizer, with all per-stream state kept on the OnlineStream. RunVad()
// runs one batched inference per round over the streams holding a full
// window.

#ifndef SHERPA_ONNX_CSRC_ONLINE_SILERO_VAD_H_
#define SHERPA_ONNX_CSRC_ONLINE_SILERO_VAD_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/online-vad-model-config.h"
#include "sherpa-onnx/csrc/online-vad-model.h"

namespace sherpa_onnx {

class OnlineSileroVad : public OnlineVadModel {
 public:
  explicit OnlineSileroVad(const OnlineVadModelConfig &config);
  ~OnlineSileroVad() override;

  void RunVad(OnlineStream **ss, int32_t n) override;
  void WarmUp(int32_t n) override;
  const char *Name() const override { return "silero"; }
  int32_t SampleRate() const override { return config_.sample_rate; }
  int32_t WindowShift() const override { return window_shift_; }

  int32_t WindowSize() const override { return window_size_; }

 private:
  void Init();

  // Fresh per-stream LSTM init state (v4: h,c [2,1,64]; v5: state [2,1,128]).
  std::vector<Ort::Value> GetInitStates();

  // Stack per-stream states [2,1,H] into [2,B,H], and split them back.
  std::vector<Ort::Value> StackStates(const std::vector<int32_t> &idx,
                                      OnlineStream **ss);
  void UnStackStates(std::vector<Ort::Value> &stacked,
                     const std::vector<int32_t> &idx, OnlineStream **ss);

  // One batched forward: x [B, window_size_] -> probs[B], new state.
  std::vector<Ort::Value> RunBatch(const float *x, int32_t batch_size,
                                   std::vector<Ort::Value> &stacked_state,
                                   std::vector<float> *probs,
                                   const Ort::MemoryInfo &memory_info);

  // Threshold / min-speech / min-silence rule, same as
  // SileroVadModel::IsSpeech, over per-stream counters.
  bool Decide(float prob, bool *triggered, int32_t *current_sample,
              int32_t *temp_start, int32_t *temp_end) const;

  OnlineVadModelConfig config_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t window_size_ = 0;     // input length (window + overlap)
  int32_t window_shift_ = 0;    // stride (= config_.window_size)
  int32_t window_overlap_ = 0;  // v5 only
  bool is_v5_ = false;
  int32_t min_silence_samples_ = 0;
  int32_t min_speech_samples_ = 0;
  float neg_threshold_ = 0.35f;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SILERO_VAD_H_
