// sherpa-onnx/csrc/smart-turn-detector.cc
//
// Copyright (c) 2026 Xiaomi Corporation

#include "sherpa-onnx/csrc/smart-turn-detector.h"

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/ort-env.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

class SmartTurnDetector::Impl {
 public:
  Impl(const SmartTurnConfig &config, int32_t sample_rate, int32_t num_threads,
       const std::string &provider, bool debug)
      : config_(config),
        sample_rate_(sample_rate),
        env_(CreateOrtEnv()) {
    auto sess_opts = GetSessionOptions(VadModelConfig({}, {}, sample_rate, num_threads, provider, debug));
    sess_ = std::make_unique<Ort::Session>(env_, SHERPA_ONNX_TO_ORT_PATH(config_.model), sess_opts);
    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
  }

  float Compute(const float *samples, int32_t n) const {
    const int32_t used_samples = static_cast<int32_t>(config_.window_size * sample_rate_);
    if (n > used_samples) {
      samples += n - used_samples;
      n = used_samples;
    }

    const int32_t expected_samples = static_cast<int32_t>(config_.window_size * config_.sample_rate);
    std::vector<float> waveform(expected_samples, 0), resample_buff;
    if (sample_rate_ != config_.sample_rate) {
      float min_freq = std::min(sample_rate_, config_.sample_rate);
      float lowpass_cutoff = 0.99 * 0.5 * min_freq;
      LinearResample resampler(sample_rate_, config_.sample_rate, lowpass_cutoff, 6);
      resampler.Resample(samples, n, true, &resample_buff);
      samples = &resample_buff[0];
      n = static_cast<int32_t>(resample_buff.size());
    }
    std::copy(samples, samples + n, &waveform[expected_samples - n]);

    const int64_t shape[] = {1, expected_samples};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory_info, 
        waveform.data(), waveform.size(), shape, 2);
    auto output = sess_->Run({}, input_names_ptr_.data(), &input, 1,
                             output_names_ptr_.data(), output_names_ptr_.size());
    return output[0].GetTensorData<float>()[0];
  }

  bool IsEndOfTurn(const float *samples, int32_t n) const {
    return Compute(samples, n) >= config_.threshold;
  }

 private:
  SmartTurnConfig config_;
  int32_t sample_rate_;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> sess_;
  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

SmartTurnDetector::SmartTurnDetector(const SmartTurnConfig &config,
                                     int32_t sample_rate, int32_t num_threads,
                                     const std::string &provider, bool debug)
    : impl_(std::make_unique<Impl>(config, sample_rate, num_threads, provider,
                                  debug)) {}

SmartTurnDetector::~SmartTurnDetector() = default;

float SmartTurnDetector::Compute(const float *samples, int32_t n) const {
  return impl_->Compute(samples, n);
}

bool SmartTurnDetector::IsEndOfTurn(const float *samples, int32_t n) const {
  return impl_->IsEndOfTurn(samples, n);
}

}  // namespace sherpa_onnx