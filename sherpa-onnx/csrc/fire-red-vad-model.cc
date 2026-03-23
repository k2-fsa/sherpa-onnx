// sherpa-onnx/csrc/fire-red-vad-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/fire-red-vad-model.h"

#include <algorithm>
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
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

class FireRedVadModel::Impl {
 public:
  explicit Impl(const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(config.fire_red_vad.model);
    Init(buf.data(), buf.size());

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config.fire_red_vad.window_size != 400) {
      SHERPA_ONNX_LOGE("Expected window size 400. Given: %d",
                       config.fire_red_vad.window_size);
      SHERPA_ONNX_EXIT(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.fire_red_vad.min_silence_duration;

    min_speech_samples_ =
        sample_rate_ * config_.fire_red_vad.min_speech_duration;
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.fire_red_vad.model);
    Init(buf.data(), buf.size());

    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config.fire_red_vad.window_size != 400) {
      SHERPA_ONNX_LOGE("Expected window size 400. Given: %d",
                       config.fire_red_vad.window_size);
      SHERPA_ONNX_EXIT(-1);
    }

    min_silence_samples_ =
        sample_rate_ * config_.fire_red_vad.min_silence_duration;

    min_speech_samples_ =
        sample_rate_ * config_.fire_red_vad.min_speech_duration;
  }

  void Reset() {
    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;

    ResetStates();
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }
    stream_->AcceptWaveform(16000, samples, n);
    if (!IsReady()) {
      return false;
    }

    int32_t &num_processed_frames = stream_->GetNumProcessedFrames();
    std::vector<float> features = stream_->GetFrames(num_processed_frames, 1);

    num_processed_frames += 1;

    float prob = Run(features.data(), features.size());

    float threshold = config_.fire_red_vad.threshold;

    current_sample_ += config_.fire_red_vad.window_size;

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

    float neg_threshold;
    if (config_.fire_red_vad.neg_threshold < 0) {
      neg_threshold = std::max(threshold - 0.15f, 0.01f);
    } else {
      neg_threshold = std::max(config_.fire_red_vad.neg_threshold, 0.01f);
    }
    if ((prob > neg_threshold) && triggered_) {
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

  int32_t WindowShift() const { return config_.fire_red_vad.window_size; }

  int32_t WindowSize() const {
    return config_.fire_red_vad.window_size + window_overlap_;
  }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
  }

  void SetThreshold(float threshold) {
    config_.fire_red_vad.threshold = threshold;
  }

  float RunFromSamples(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }

    stream_->AcceptWaveform(16000, samples, n);
    if (!IsReady()) {
      return 0;
    }

    int32_t &num_processed_frames = stream_->GetNumProcessedFrames();
    std::vector<float> features = stream_->GetFrames(num_processed_frames, 1);

    num_processed_frames += 1;

    return Run(features.data(), features.size());
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(model_type, "model_type");
    if (model_type != "fire-red-vad") {
      SHERPA_ONNX_LOGE("Expect model type fire-red-vad. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_LOGE(
          "Please download FireRedVad from sherpa-onnx. DON'T use models from "
          "elsewhere");
      SHERPA_ONNX_EXIT(-1);
    }

    Reset();

    FeatureExtractorConfig feat_config;
    feat_config.normalize_samples = false;
    feat_config.snip_edges = true;

    stream_ = std::make_unique<OnlineStream>(feat_config);
  }

  void ResetStates() {
    std::array<int64_t, 3> shape{1, 128, 0};

    states_.clear();

    for (int32_t i = 0; i != 8; ++i) {
      Ort::Value s = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                                     shape.size());

      states_.push_back(std::move(s));
    }
  }

  bool IsReady() const {
    return stream_->GetNumProcessedFrames() + 1 < stream_->NumFramesReady();
  }

  float Run(const float *features, int32_t n) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape = {1, 1, n};

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(features), n,
                                 x_shape.data(), x_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.reserve(input_names_.size());

    inputs.push_back(std::move(x));

    for (auto &s : states_) {
      inputs.push_back(std::move(s));
    }

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    for (int32_t i = 0; i != static_cast<int32_t>(states_.size()); ++i) {
      states_[i] = std::move(out[i + 1]);
    }

    float prob = out[0].GetTensorData<float>()[0];
    return prob;
  }

 private:
  VadModelConfig config_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  std::vector<Ort::Value> states_;
  int64_t sample_rate_;
  int32_t min_silence_samples_;
  int32_t min_speech_samples_;

  bool triggered_ = false;
  int32_t current_sample_ = 0;
  int32_t temp_start_ = 0;
  int32_t temp_end_ = 0;

  int32_t window_overlap_ = 0;

  std::unique_ptr<OnlineStream> stream_;
};

FireRedVadModel::FireRedVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
FireRedVadModel::FireRedVadModel(Manager *mgr, const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

FireRedVadModel::~FireRedVadModel() = default;

void FireRedVadModel::Reset() { return impl_->Reset(); }

bool FireRedVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t FireRedVadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t FireRedVadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t FireRedVadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t FireRedVadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void FireRedVadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void FireRedVadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

float FireRedVadModel::Compute(const float *samples, int32_t n) {
  return impl_->RunFromSamples(samples, n);
}

#if __ANDROID_API__ >= 9
template FireRedVadModel::FireRedVadModel(AAssetManager *mgr,
                                          const VadModelConfig &config);
#endif

#if __OHOS__
template FireRedVadModel::FireRedVadModel(NativeResourceManager *mgr,
                                          const VadModelConfig &config);
#endif

}  // namespace sherpa_onnx
