// sherpa-onnx/csrc/rknn/silero-vad-model-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/silero-vad-model-rknn.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/rknn/macros.h"
#include "sherpa-onnx/csrc/rknn/utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class SileroVadModelRknn::Impl {
 public:
  ~Impl() {
    auto ret = rknn_destroy(ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the context");
    }
  }

  explicit Impl(const VadModelConfig &config)
      : config_(config), sample_rate_(config.sample_rate) {
    auto buf = ReadFile(config.silero_vad.model);
    Init(buf.data(), buf.size());

    SetCoreMask(ctx_, config_.num_threads);

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.silero_vad.min_silence_duration;

    min_speech_samples_ = sample_rate_ * config_.silero_vad.min_speech_duration;
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config), sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.silero_vad.model);
    Init(buf.data(), buf.size());

    SetCoreMask(ctx_, config_.num_threads);

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      exit(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.silero_vad.min_silence_duration;

    min_speech_samples_ = sample_rate_ * config_.silero_vad.min_speech_duration;
  }

  void Reset() {
    for (auto &s : states_) {
      std::fill(s.begin(), s.end(), 0);
    }

    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }

    float prob = Run(samples, n);

    float threshold = config_.silero_vad.threshold;

    current_sample_ += config_.silero_vad.window_size;

    if (prob > threshold && temp_end_ != 0) {
      temp_end_ = 0;
    }

    if (prob > threshold && temp_start_ == 0) {
      // start speaking, but we require that it must satisfy
      // min_speech_duration
      temp_start_ = current_sample_;
      return false;
    }

    if (prob > threshold && temp_start_ != 0 && !triggered_) {
      if (current_sample_ - temp_start_ < min_speech_samples_) {
        return false;
      }

      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && !triggered_) {
      // silence
      temp_start_ = 0;
      temp_end_ = 0;
      return false;
    }

    if ((prob > threshold - 0.15) && triggered_) {
      // speaking
      return true;
    }

    if ((prob > threshold) && !triggered_) {
      // start speaking
      triggered_ = true;

      return true;
    }

    if ((prob < threshold) && triggered_) {
      // stop to speak
      if (temp_end_ == 0) {
        temp_end_ = current_sample_;
      }

      if (current_sample_ - temp_end_ < min_silence_samples_) {
        // continue speaking
        return true;
      }
      // stopped speaking
      temp_start_ = 0;
      temp_end_ = 0;
      triggered_ = false;
      return false;
    }

    return false;
  }

  int32_t WindowShift() const { return config_.silero_vad.window_size; }

  int32_t WindowSize() const {
    return config_.silero_vad.window_size + window_overlap_;
  }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
  }

  void SetThreshold(float threshold) {
    config_.silero_vad.threshold = threshold;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &ctx_);

    InitInputOutputAttrs(ctx_, config_.debug, &input_attrs_, &output_attrs_);

    rknn_custom_string custom_string = GetCustomString(ctx_, config_.debug);

    auto meta = Parse(custom_string, config_.debug);

    if (config_.silero_vad.window_size != 512) {
      SHERPA_ONNX_LOGE("we require window_size to be 512. Given: %d",
                       config_.silero_vad.window_size);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      for (const auto &p : meta) {
        SHERPA_ONNX_LOGE("%s: %s", p.first.c_str(), p.second.c_str());
      }
    }

    if (meta.count("model_type") == 0) {
      SHERPA_ONNX_LOGE("No model type found in '%s'",
                       config_.silero_vad.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.at("model_type") != "silero-vad-v4") {
      SHERPA_ONNX_LOGE("Expect model type silero-vad-v4 in '%s', given: '%s'",
                       config_.silero_vad.model.c_str(),
                       meta.at("model_type").c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.count("sample_rate") == 0) {
      SHERPA_ONNX_LOGE("No sample_rate found in '%s'",
                       config_.silero_vad.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.at("sample_rate") != "16000") {
      SHERPA_ONNX_LOGE("Expect sample rate 16000 in '%s', given: '%s'",
                       config_.silero_vad.model.c_str(),
                       meta.at("sample_rate").c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.count("version") == 0) {
      SHERPA_ONNX_LOGE("No version found in '%s'",
                       config_.silero_vad.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.at("version") != "4") {
      SHERPA_ONNX_LOGE("Expect version 4 in '%s', given: '%s'",
                       config_.silero_vad.model.c_str(),
                       meta.at("version").c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.count("h_shape") == 0) {
      SHERPA_ONNX_LOGE("No h_shape found in '%s'",
                       config_.silero_vad.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta.count("c_shape") == 0) {
      SHERPA_ONNX_LOGE("No c_shape found in '%s'",
                       config_.silero_vad.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int64_t> h_shape;
    std::vector<int64_t> c_shape;

    SplitStringToIntegers(meta.at("h_shape"), ",", false, &h_shape);
    SplitStringToIntegers(meta.at("c_shape"), ",", false, &c_shape);
    if (h_shape.size() != 3 || c_shape.size() != 3) {
      SHERPA_ONNX_LOGE("Incorrect shape for h (%d) or c (%d)",
                       static_cast<int32_t>(h_shape.size()),
                       static_cast<int32_t>(c_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    states_.resize(2);
    states_[0].resize(h_shape[0] * h_shape[1] * h_shape[2]);
    states_[1].resize(c_shape[0] * c_shape[1] * c_shape[2]);

    Reset();
  }

  float Run(const float *samples, int32_t n) {
    std::vector<rknn_input> inputs(input_attrs_.size());

    for (int32_t i = 0; i < static_cast<int32_t>(inputs.size()); ++i) {
      auto &input = inputs[i];
      auto &attr = input_attrs_[i];
      input.index = attr.index;

      if (attr.type == RKNN_TENSOR_FLOAT16) {
        input.type = RKNN_TENSOR_FLOAT32;
      } else if (attr.type == RKNN_TENSOR_INT64) {
        input.type = RKNN_TENSOR_INT64;
      } else {
        SHERPA_ONNX_LOGE("Unsupported tensor type %d, %s", attr.type,
                         get_type_string(attr.type));
        SHERPA_ONNX_EXIT(-1);
      }

      input.fmt = attr.fmt;
      if (i == 0) {
        input.buf = reinterpret_cast<void *>(const_cast<float *>(samples));
        input.size = n * sizeof(float);
      } else {
        input.buf = reinterpret_cast<void *>(states_[i - 1].data());
        input.size = states_[i - 1].size() * sizeof(float);
      }
    }

    std::vector<float> out(output_attrs_[0].n_elems);

    auto &next_states = states_;

    std::vector<rknn_output> outputs(output_attrs_.size());

    for (int32_t i = 0; i < outputs.size(); ++i) {
      auto &output = outputs[i];
      auto &attr = output_attrs_[i];
      output.index = attr.index;
      output.is_prealloc = 1;

      if (attr.type == RKNN_TENSOR_FLOAT16) {
        output.want_float = 1;
      } else if (attr.type == RKNN_TENSOR_INT64) {
        output.want_float = 0;
      } else {
        SHERPA_ONNX_LOGE("Unsupported tensor type %d, %s", attr.type,
                         get_type_string(attr.type));
        SHERPA_ONNX_EXIT(-1);
      }

      if (i == 0) {
        output.size = out.size() * sizeof(float);
        output.buf = reinterpret_cast<void *>(out.data());
      } else {
        output.size = next_states[i - 1].size() * sizeof(float);
        output.buf = reinterpret_cast<void *>(next_states[i - 1].data());
      }
    }

    auto ret = rknn_inputs_set(ctx_, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set inputs");

    ret = rknn_run(ctx_, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run the model");

    ret = rknn_outputs_get(ctx_, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get model output");

    return out[0];
  }

 private:
  VadModelConfig config_;
  rknn_context ctx_ = 0;

  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;

  std::vector<std::vector<float>> states_;

  int64_t sample_rate_;
  int32_t min_silence_samples_;
  int32_t min_speech_samples_;

  bool triggered_ = false;
  int32_t current_sample_ = 0;
  int32_t temp_start_ = 0;
  int32_t temp_end_ = 0;

  int32_t window_overlap_ = 0;
};

SileroVadModelRknn::SileroVadModelRknn(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
SileroVadModelRknn::SileroVadModelRknn(Manager *mgr,
                                       const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

SileroVadModelRknn::~SileroVadModelRknn() = default;

void SileroVadModelRknn::Reset() { return impl_->Reset(); }

bool SileroVadModelRknn::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t SileroVadModelRknn::WindowSize() const { return impl_->WindowSize(); }

int32_t SileroVadModelRknn::WindowShift() const { return impl_->WindowShift(); }

int32_t SileroVadModelRknn::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t SileroVadModelRknn::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void SileroVadModelRknn::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void SileroVadModelRknn::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

#if __ANDROID_API__ >= 9
template SileroVadModelRknn::SileroVadModelRknn(AAssetManager *mgr,
                                                const VadModelConfig &config);
#endif

#if __OHOS__
template SileroVadModelRknn::SileroVadModelRknn(NativeResourceManager *mgr,
                                                const VadModelConfig &config);
#endif

}  // namespace sherpa_onnx
