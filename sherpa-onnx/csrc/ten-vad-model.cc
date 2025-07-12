// sherpa-onnx/csrc/ten-vad-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/ten-vad-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/rfft.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class TenVadModel::Impl {
 public:
  explicit Impl(const VadModelConfig &config)
      : config_(config),
        rfft_(1024),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(config.ten_vad.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config),
        rfft_(1024),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.ten_vad.model);
    Init(buf.data(), buf.size());
  }

  void Reset() {
    triggered_ = false;
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;

    last_sample_ = 0;

    last_features_.resize(3 * 41);
    std::fill(last_features_.begin(), last_features_.end(), 0.0f);
    tmp_samples_.resize(1024);

    ResetStates();
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }

    float prob = Run(samples, n);

    float threshold = config_.ten_vad.threshold;

    current_sample_ += config_.ten_vad.window_size;

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

  int32_t WindowShift() const { return config_.ten_vad.window_size; }

  int32_t WindowSize() const { return config_.ten_vad.window_size; }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
  }

  void SetThreshold(float threshold) { config_.ten_vad.threshold = threshold; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config_.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.ten_vad.window_size > 768) {
      SHERPA_ONNX_LOGE("Windows size %d for ten-vad is too large",
                       config_.ten_vad.window_size);
      SHERPA_ONNX_EXIT(-1);
    }

    min_silence_samples_ = sample_rate_ * config_.ten_vad.min_silence_duration;

    min_speech_samples_ = sample_rate_ * config_.ten_vad.min_speech_duration;

    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    InitMelBanks();

    Check();

    Reset();
  }

  void ResetStates() {
    std::array<int64_t, 2> shape{1, 64};

    states_.clear();
    states_.reserve(4);
    for (int32_t i = 0; i != 4; ++i) {
      Ort::Value s = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                                     shape.size());

      Fill<float>(&s, 0);
      states_.push_back(std::move(s));
    }
  }

  void InitMelBanks() {
    knf::FrameExtractionOptions frame_opts;

    // 16 kHz, so num_fft is 16000*64/1000 = 1024
    frame_opts.frame_length_ms = 64;

    knf::MelBanksOptions mel_opts;
    mel_opts.is_librosa = true;
    mel_opts.norm = "";
    mel_opts.use_slaney_mel_scale = true;
    mel_opts.floor_to_int_bin = true;
    mel_opts.low_freq = 0;
    mel_opts.high_freq = 8000;
    mel_opts.num_bins = 40;

    mel_banks_ = std::make_unique<knf::MelBanks>(mel_opts, frame_opts, 1.0f);

    features_.resize(41);
  }

  void Check() {
    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---ten-vad---\n";
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(model_type, "model_type");

    if (model_type.empty()) {
      SHERPA_ONNX_LOGE(
          "Please download ten-vad.onnx or ten-vad.int8.onnx from\n"
          "https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models"
          "\nWe have added meta data to the original ten-vad.onnx from\n"
          "https://github.com/TEN-framework/ten-vad");
      SHERPA_ONNX_EXIT(-1);
    }

    if (model_type != "ten-vad") {
      SHERPA_ONNX_LOGE("Expect model type 'ten-vad', given '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(mean_, "mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(inv_stddev_, "inv_stddev");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(window_, "window");

    if (mean_.size() != 41) {
      SHERPA_ONNX_LOGE(
          "Incorrect size of the mean vector. Given %d, expected 41",
          static_cast<int32_t>(mean_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (inv_stddev_.size() != 41) {
      SHERPA_ONNX_LOGE(
          "Incorrect size of the inv_stddev vector. Given %d, expected 41",
          static_cast<int32_t>(inv_stddev_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (window_.size() != 768) {
      SHERPA_ONNX_LOGE(
          "Incorrect size of the window vector. Given %d, expected 768",
          static_cast<int32_t>(window_.size()));
      SHERPA_ONNX_EXIT(-1);
    }
  }

  static void Scale(const float *samples, int32_t n, float *out) {
    for (int32_t i = 0; i != n; ++i) {
      out[i] = samples[i] * 32768;
    }
  }

  void Preemphasis(const float *samples, int32_t n, float *out) {
    float t = samples[n - 1];

    for (int32_t i = n - 1; i > 0; --i) {
      out[i] = samples[i] - 0.97 * samples[i - 1];
    }

    out[0] = samples[0] - 0.97 * last_sample_;

    last_sample_ = t;
  }

  static void ApplyWindow(const float *samples, const float *window, int32_t n,
                          float *out) {
    for (int32_t i = 0; i != n; ++i) {
      out[i] = samples[i] * window[i];
    }
  }

  static void ComputePowerSpectrum(const float *fft_bins, int32_t n,
                                   float *out) {
    out[0] = fft_bins[0] * fft_bins[0];
    out[n - 1] = fft_bins[1] * fft_bins[1];

    for (int32_t i = 1; i < n / 2; ++i) {
      float real = fft_bins[2 * i];
      float imag = fft_bins[2 * i + 1];
      out[i] = real * real + imag * imag;
    }
  }

  static void LogMel(const float *in, int32_t n, float *out) {
    for (int32_t i = 0; i != n; ++i) {
      // 20.79441541679836 is log(32768*32768)
      out[i] = logf(in[i] + 1e-10f) - 20.79441541679836f;
    }
  }

  void ApplyNormalization(const float *in, float *out) const {
    for (int32_t i = 0; i != static_cast<int32_t>(mean_.size()); ++i) {
      out[i] = (in[i] - mean_[i]) * inv_stddev_[i];
    }
  }

  void ComputeFeatures(const float *samples, int32_t n) {
    std::fill(tmp_samples_.begin() + n, tmp_samples_.end(), 0.0f);

    Scale(samples, n, tmp_samples_.data());

    Preemphasis(tmp_samples_.data(), n, tmp_samples_.data());
    ApplyWindow(tmp_samples_.data(), window_.data(), n, tmp_samples_.data());

    rfft_.Compute(tmp_samples_.data());
    auto &power_spectrum = tmp_samples_;
    ComputePowerSpectrum(tmp_samples_.data(), tmp_samples_.size(),
                         power_spectrum.data());

    // note only the first half of power_spectrum is used inside Compute()
    mel_banks_->Compute(power_spectrum.data(), features_.data());
    LogMel(features_.data(), static_cast<int32_t>(features_.size()) - 1,
           features_.data());

    // Note(fangjun): The ten-vad model expects a pitch feature, but we set it
    // to 0 as a simplification. This may reduce performance as noted
    // in the PR #2377
    features_.back() = 0;

    ApplyNormalization(features_.data(), features_.data());

    std::memmove(last_features_.data(),
                 last_features_.data() + features_.size(),
                 2 * features_.size() * sizeof(float));
    std::copy(features_.begin(), features_.end(),
              last_features_.begin() + 2 * features_.size());
  }

  float Run(const float *samples, int32_t n) {
    ComputeFeatures(samples, n);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape = {1, 3, 41};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, last_features_.data(),
                                            last_features_.size(),
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

    for (int32_t i = 1; i != static_cast<int32_t>(output_names_.size()); ++i) {
      states_[i - 1] = std::move(out[i]);
    }

    float prob = out[0].GetTensorData<float>()[0];

    return prob;
  }

 private:
  VadModelConfig config_;
  knf::Rfft rfft_;
  std::unique_ptr<knf::MelBanks> mel_banks_;

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

  float last_sample_ = 0;

  std::vector<float> mean_;
  std::vector<float> inv_stddev_;
  std::vector<float> window_;

  std::vector<float> features_;
  std::vector<float> last_features_;  // (3, 41), row major
  std::vector<float> tmp_samples_;    // (1024,)
};

TenVadModel::TenVadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
TenVadModel::TenVadModel(Manager *mgr, const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

TenVadModel::~TenVadModel() = default;

void TenVadModel::Reset() { return impl_->Reset(); }

bool TenVadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

int32_t TenVadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t TenVadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t TenVadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t TenVadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void TenVadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void TenVadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

#if __ANDROID_API__ >= 9
template TenVadModel::TenVadModel(AAssetManager *mgr,
                                  const VadModelConfig &config);
#endif

#if __OHOS__
template TenVadModel::TenVadModel(NativeResourceManager *mgr,
                                  const VadModelConfig &config);
#endif

}  // namespace sherpa_onnx
