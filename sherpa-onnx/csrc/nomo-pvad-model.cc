// sherpa-onnx/csrc/nomo-pvad-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/nomo-pvad-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
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

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class NomoPvadModel::Impl {
 public:
  explicit Impl(const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.nomo_pvad.model), sess_opts_);
    Init(nullptr, 0, [](const std::string &p) { return ReadFile(p); });
  }

  template <typename Manager>
  Impl(Manager *mgr, const VadModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        sample_rate_(config.sample_rate) {
    auto buf = ReadFile(mgr, config.nomo_pvad.model);
    Init(buf.data(), buf.size(),
         [mgr](const std::string &p) { return ReadFile(mgr, p); });
  }

  void Reset() {
    current_sample_ = 0;
    temp_start_ = 0;
    temp_end_ = 0;
    triggered_ = false;

    keep_ = 0;
    oldest_ = 0;
    buffer_.clear();
  }

  bool IsSpeech(const float *samples, int32_t n) {
    if (n != WindowSize()) {
      SHERPA_ONNX_LOGE("n: %d != window_size: %d", n, WindowSize());
      SHERPA_ONNX_EXIT(-1);
    }

    float prob = Run(samples, n);

    return UpdateState(prob);
  }

  float Compute(const float *samples, int32_t n) { return Run(samples, n); }

  int32_t WindowSize() const { return chunk_samples_; }

  int32_t WindowShift() const { return chunk_samples_; }

  int32_t MinSilenceDurationSamples() const { return min_silence_samples_; }

  int32_t MinSpeechDurationSamples() const { return min_speech_samples_; }

  void SetMinSilenceDuration(float s) {
    min_silence_samples_ = sample_rate_ * s;
  }

  void SetThreshold(float threshold) {
    config_.nomo_pvad.threshold = threshold;
    float neg = config_.nomo_pvad.neg_threshold;
    if (neg < 0) {
      neg_threshold_ = std::max(threshold - 0.15f, 0.01f);
    } else {
      neg_threshold_ = neg;
    }
  }

 private:
  using FileReader = std::function<std::vector<char>(const std::string &)>;

  void Init(void *model_data, size_t model_data_length,
            const FileReader &reader) {
    if (sample_rate_ != 16000) {
      SHERPA_ONNX_LOGE("Expected sample rate 16000. Given: %d",
                       config_.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }

    if (model_data) {
      sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                             model_data_length, sess_opts_);
    } else if (!sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Check(reader);

    min_silence_samples_ =
        sample_rate_ * config_.nomo_pvad.min_silence_duration;
    min_speech_samples_ = sample_rate_ * config_.nomo_pvad.min_speech_duration;

    SetThreshold(config_.nomo_pvad.threshold);

    Reset();
  }

  void Check(const FileReader &reader) {
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---nomo-pvad---\n";
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
          "Please use a nomo-pvad.onnx that contains the meta data read by "
          "sherpa-onnx.\nSee "
          "https://k2-fsa.github.io/sherpa/onnx/vad/nomo-pvad.html\n"
          "for the downloadable model files and for how to export them "
          "from https://huggingface.co/TuyaNomo/nomo-pvad");
      SHERPA_ONNX_EXIT(-1);
    }

    if (model_type != "nomo-pvad") {
      SHERPA_ONNX_LOGE("Expect model type 'nomo-pvad', given '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t model_sample_rate = -1;
    SHERPA_ONNX_READ_META_DATA(model_sample_rate, "sample_rate");
    if (model_sample_rate != sample_rate_) {
      SHERPA_ONNX_LOGE("Expected sample rate %d. Given: %d", model_sample_rate,
                       sample_rate_);
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(feature_dim_, "feature_dim");
    SHERPA_ONNX_READ_META_DATA(chunk_samples_, "chunk_samples");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(window_samples_, "window_samples",
                                            38400);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(spk_embed_dim_, "spk_embed_dim",
                                            192);

    if (feature_dim_ != 80) {
      SHERPA_ONNX_LOGE("Expected feature dim 80. Given: %d", feature_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (chunk_samples_ <= 0) {
      SHERPA_ONNX_LOGE("Invalid chunk_samples: %d", chunk_samples_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (window_samples_ < chunk_samples_) {
      SHERPA_ONNX_LOGE(
          "window_samples (%d) must not be smaller than chunk_samples (%d)",
          window_samples_, chunk_samples_);
      SHERPA_ONNX_EXIT(-1);
    }

    std::string feature_normalize_type;
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(feature_normalize_type,
                                                "feature_normalize_type", "");
    if (!feature_normalize_type.empty() &&
        feature_normalize_type != "global-mean") {
      SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %s",
                       feature_normalize_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    subtract_global_mean_ = feature_normalize_type == "global-mean";

    ResolveTargetSpeakerEmbedding(reader);

    // Check the order of the model inputs so that we can fill them in the
    // right order below.
    spk_emb_index_ = -1;
    fbank_index_ = -1;
    for (int32_t i = 0; i != static_cast<int32_t>(input_names_.size()); ++i) {
      if (input_names_[i] == "spk_emb") {
        spk_emb_index_ = i;
      } else if (input_names_[i] == "fbank") {
        fbank_index_ = i;
      }
    }

    if (spk_emb_index_ < 0 || fbank_index_ < 0) {
      SHERPA_ONNX_LOGE(
          "Expect the model to have 'spk_emb' and 'fbank' as inputs. Given:");
      for (const auto &n : input_names_) {
        SHERPA_ONNX_LOGE("  %s", n.c_str());
      }
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void ResolveTargetSpeakerEmbedding(const FileReader &reader) {
    const auto &cfg = config_.nomo_pvad;

    if (!cfg.speaker_embedding.empty()) {
      spk_emb_ = cfg.speaker_embedding;
    } else if (!cfg.speaker_embedding_file.empty()) {
      auto buf = reader(cfg.speaker_embedding_file);
      std::string text(buf.begin(), buf.end());
      if (!SplitStringToFloats(text, ", \t\r\n", /*omit_empty*/ true,
                               &spk_emb_)) {
        SHERPA_ONNX_LOGE("Failed to parse the enrollment embedding from '%s'",
                         cfg.speaker_embedding_file.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    } else {
      SHERPA_ONNX_LOGE(
          "Please provide the enrollment embedding of the target speaker");
      SHERPA_ONNX_EXIT(-1);
    }

    if (static_cast<int32_t>(spk_emb_.size()) != spk_embed_dim_) {
      SHERPA_ONNX_LOGE(
          "Expected a %d-d enrollment embedding. Given: %d. Please use the "
          "same speaker embedding model that nomo-pvad was trained with, "
          "i.e., 3d-speaker ERes2NetV2",
          spk_embed_dim_, static_cast<int32_t>(spk_emb_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    // The FiLM layer starts with a LayerNorm, so the model is scale
    // invariant. Normalizing anyway keeps the numbers well behaved.
    float sum = 0;
    for (float f : spk_emb_) {
      sum += f * f;
    }

    float norm = std::sqrt(sum);
    if (norm <= 0) {
      SHERPA_ONNX_LOGE("The enrollment embedding is all zeros");
      SHERPA_ONNX_EXIT(-1);
    }

    for (auto &f : spk_emb_) {
      f /= norm;
    }
  }

  // Append n samples to the audio history and return the probability that
  // the target speaker is talking in the newest chunk.
  float Run(const float *samples, int32_t n) {
    if (n <= 0) {
      return 0.0f;
    }

    AppendSamples(samples, n);

    return RunOnWindow();
  }

  void AppendSamples(const float *samples, int32_t n) {
    // `buffer_` is a ring buffer of size `window_samples_` holding the most
    // recent `keep_` samples; `oldest_` is the index of the oldest one.
    if (static_cast<int32_t>(buffer_.size()) != window_samples_) {
      buffer_.resize(window_samples_, 0.0f);
      keep_ = 0;
      oldest_ = 0;
    }

    for (int32_t i = 0; i != n; ++i) {
      int32_t idx = oldest_ + keep_;
      if (idx >= window_samples_) {
        idx -= window_samples_;
      }

      buffer_[idx] = samples[i];

      if (keep_ < window_samples_) {
        ++keep_;
      } else {
        ++oldest_;
        if (oldest_ >= window_samples_) {
          oldest_ = 0;
        }
      }
    }
  }

  float RunOnWindow() {
    if (keep_ <= 0) {
      return 0.0f;
    }

    // Linearize the ring buffer, oldest sample first.
    std::vector<float> waveform(keep_);
    int32_t first = std::min(keep_, window_samples_ - oldest_);
    std::copy(buffer_.begin() + oldest_, buffer_.begin() + oldest_ + first,
              waveform.begin());
    if (keep_ > first) {
      std::copy(buffer_.begin(), buffer_.begin() + (keep_ - first),
                waveform.begin() + first);
    }

    // Compute features for the whole window. A fresh extractor is used
    // because every window is scored independently.
    FeatureExtractorConfig feat_config;
    feat_config.sampling_rate = sample_rate_;
    feat_config.feature_dim = feature_dim_;
    feat_config.normalize_samples = true;
    // The model was trained with torchaudio.compliance.kaldi.fbank, whose
    // defaults differ from FeatureExtractorConfig in two places:
    //   - snip_edges is True
    //   - high_freq is 0, i.e., Nyquist, instead of Nyquist - 400
    feat_config.snip_edges = true;
    feat_config.high_freq = 0.5f * sample_rate_;

    FeatureExtractor extractor(feat_config);
    extractor.AcceptWaveform(sample_rate_, waveform.data(),
                             static_cast<int32_t>(waveform.size()));

    int32_t num_frames = extractor.NumFramesReady();
    if (num_frames <= 0) {
      return 0.0f;
    }

    std::vector<float> features = extractor.GetFrames(0, num_frames);

    if (subtract_global_mean_) {
      SubtractGlobalMean(features.data(), num_frames, feature_dim_);
    }

    // The ERes2NetV2 trunk downsamples time by 8 and the chunk head averages
    // every 2 trunk frames, so the number of fbank frames must be a multiple
    // of 16. Zero-pad the tail, exactly like the reference implementation.
    int32_t num_chunks = (num_frames + 15) / 16;
    int32_t num_padded_frames = num_chunks * 16;
    features.resize(static_cast<size_t>(num_padded_frames) * feature_dim_,
                    0.0f);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> fbank_shape{1, num_padded_frames, feature_dim_};
    Ort::Value fbank = Ort::Value::CreateTensor<float>(
        memory_info, features.data(), features.size(), fbank_shape.data(),
        fbank_shape.size());

    std::array<int64_t, 2> emb_shape{1, spk_embed_dim_};
    Ort::Value spk_emb = Ort::Value::CreateTensor<float>(
        memory_info, spk_emb_.data(), spk_emb_.size(), emb_shape.data(),
        emb_shape.size());

    std::vector<Ort::Value> inputs(input_names_.size());
    inputs[spk_emb_index_] = std::move(spk_emb);
    inputs[fbank_index_] = std::move(fbank);

    auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                          inputs.size(), output_names_ptr_.data(),
                          output_names_ptr_.size());

    const float *p = out[0].GetTensorData<float>();
    auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t num_chunks_in_output = shape[1];

    if (num_chunks_in_output < 1) {
      return 0.0f;
    }

    // Class 1 is the target speaker. Use a numerically stable softmax.
    const float *last = p + (num_chunks_in_output - 1) * 2;
    float a = last[0];
    float b = last[1];
    float m = std::max(a, b);
    float ea = std::exp(a - m);
    float eb = std::exp(b - m);
    return eb / (ea + eb);
  }

  bool UpdateState(float prob) {
    float threshold = config_.nomo_pvad.threshold;

    current_sample_ += chunk_samples_;

    if (prob >= threshold) {
      temp_end_ = 0;

      if (!triggered_) {
        if (temp_start_ == 0) {
          // The target speaker might have started talking. We require that
          // the speech lasts for at least min_speech_duration.
          temp_start_ = current_sample_;
          return false;
        }

        if (current_sample_ - temp_start_ < min_speech_samples_) {
          return false;
        }

        triggered_ = true;
      }

      return true;
    }

    // prob < threshold
    if (triggered_) {
      if (prob >= neg_threshold_) {
        // Inside the hysteresis band: keep the current state.
        return true;
      }

      if (temp_end_ == 0) {
        temp_end_ = current_sample_;
      }

      if (current_sample_ - temp_end_ < min_silence_samples_) {
        // Still waiting for the silence to be long enough.
        return true;
      }

      // The target speaker stopped talking.
      temp_start_ = 0;
      temp_end_ = 0;
      triggered_ = false;
      return false;
    }

    // Not in speech and prob is low: nothing to do.
    temp_start_ = 0;
    temp_end_ = 0;
    return false;
  }

  static void SubtractGlobalMean(float *p, int32_t num_frames,
                                 int32_t feat_dim) {
    auto m = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        p, num_frames, feat_dim);

    m = m.rowwise() - m.colwise().mean();
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

  int64_t sample_rate_;

  int32_t feature_dim_ = 80;
  int32_t chunk_samples_ = 2560;
  int32_t window_samples_ = 38400;
  int32_t spk_embed_dim_ = 192;

  int32_t spk_emb_index_ = 0;
  int32_t fbank_index_ = 1;

  bool subtract_global_mean_ = true;

  std::vector<float> spk_emb_;

  // Ring buffer holding the last `window_samples_` samples.
  std::vector<float> buffer_;
  int32_t keep_ = 0;    // number of valid samples in buffer_
  int32_t oldest_ = 0;  // index of the oldest sample

  int32_t min_silence_samples_ = 0;
  int32_t min_speech_samples_ = 0;

  bool triggered_ = false;
  int64_t current_sample_ = 0;
  int64_t temp_start_ = 0;
  int64_t temp_end_ = 0;

  float neg_threshold_ = 0.55f;
};

NomoPvadModel::NomoPvadModel(const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
NomoPvadModel::NomoPvadModel(Manager *mgr, const VadModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

NomoPvadModel::~NomoPvadModel() = default;

void NomoPvadModel::Reset() { impl_->Reset(); }

bool NomoPvadModel::IsSpeech(const float *samples, int32_t n) {
  return impl_->IsSpeech(samples, n);
}

float NomoPvadModel::Compute(const float *samples, int32_t n) {
  return impl_->Compute(samples, n);
}

int32_t NomoPvadModel::WindowSize() const { return impl_->WindowSize(); }

int32_t NomoPvadModel::WindowShift() const { return impl_->WindowShift(); }

int32_t NomoPvadModel::MinSilenceDurationSamples() const {
  return impl_->MinSilenceDurationSamples();
}

int32_t NomoPvadModel::MinSpeechDurationSamples() const {
  return impl_->MinSpeechDurationSamples();
}

void NomoPvadModel::SetMinSilenceDuration(float s) {
  impl_->SetMinSilenceDuration(s);
}

void NomoPvadModel::SetThreshold(float threshold) {
  impl_->SetThreshold(threshold);
}

#if __ANDROID_API__ >= 9
template NomoPvadModel::NomoPvadModel(AAssetManager *mgr,
                                      const VadModelConfig &config);
#endif

#if __OHOS__
template NomoPvadModel::NomoPvadModel(NativeResourceManager *mgr,
                                      const VadModelConfig &config);
#endif

}  // namespace sherpa_onnx
