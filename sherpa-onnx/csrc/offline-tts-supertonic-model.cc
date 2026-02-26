// sherpa-onnx/csrc/offline-tts-supertonic-model.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-model.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include <cstring>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {
static inline bool IsCudaProvider(const std::string &provider) {
  return provider == "cuda";
}
}  // namespace

class OfflineTtsSupertonicModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    InitCudaIOBinding();
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    InitCudaIOBinding();
    Init(mgr);
  }

  const SupertonicConfig &GetConfig() const { return cfg_; }
  int32_t GetSampleRate() const { return cfg_.ae.sample_rate; }

  bool UseCudaIOBinding() const { return use_cuda_iobinding_; }
  const Ort::MemoryInfo &GetCpuMemoryInfo() const { return cpu_mem_info_; }
  const Ort::MemoryInfo *GetCudaMemoryInfo() const {
    return cuda_mem_info_.get();
  }
  std::string GetProvider() const { return config_.provider; }

  Ort::Value RunDurationPredictor(std::vector<Ort::Value> inputs) const {
    auto outputs =
        dp_sess_->Run(Ort::RunOptions{nullptr}, dp_input_names_ptr_.data(),
                      inputs.data(), inputs.size(), dp_output_names_ptr_.data(),
                      dp_output_names_ptr_.size());
    return std::move(outputs[0]);
  }

  Ort::Value RunTextEncoder(std::vector<Ort::Value> inputs) const {
    auto outputs = text_enc_sess_->Run(
        Ort::RunOptions{nullptr}, text_enc_input_names_ptr_.data(),
        inputs.data(), inputs.size(), text_enc_output_names_ptr_.data(),
        text_enc_output_names_ptr_.size());
    return std::move(outputs[0]);
  }

  Ort::Value RunVectorEstimator(std::vector<Ort::Value> inputs) const {
    auto outputs = vector_est_sess_->Run(
        Ort::RunOptions{nullptr}, vector_est_input_names_ptr_.data(),
        inputs.data(), inputs.size(), vector_est_output_names_ptr_.data(),
        vector_est_output_names_ptr_.size());
    return std::move(outputs[0]);
  }

  Ort::Value RunVocoder(std::vector<Ort::Value> inputs) const {
    auto outputs = vocoder_sess_->Run(
        Ort::RunOptions{nullptr}, vocoder_input_names_ptr_.data(),
        inputs.data(), inputs.size(), vocoder_output_names_ptr_.data(),
        vocoder_output_names_ptr_.size());
    return std::move(outputs[0]);
  }

 private:
  void PrintModelInfo(Ort::Session *sess, const std::string &name) const {
    if (!config_.debug) {
      return;
    }
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_ptr, output_names_ptr;
    GetInputNames(sess, &input_names, &input_names_ptr);
    GetOutputNames(sess, &output_names, &output_names_ptr);
    std::ostringstream os;
    os << "----------" << name << "----------\n";
    os << "Input names: ";
    for (const auto &n : input_names) os << n << " ";
    os << "\nOutput names: ";
    for (const auto &n : output_names) os << n << " ";
    os << "\n";
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
  }

  void PrintDebugInfo(const std::string &tts_config_path) const {
    if (!config_.debug) {
      return;
    }
    std::ostringstream os;
    os << "---supertonic model---\n";
    os << "tts_config: " << tts_config_path << "\n";
    os << "sample_rate: " << cfg_.ae.sample_rate << "\n";
    os << "base_chunk_size: " << cfg_.ae.base_chunk_size << "\n";
    os << "chunk_compress_factor: " << cfg_.ttl.chunk_compress_factor << "\n";
    os << "latent_dim: " << cfg_.ttl.latent_dim << "\n";
#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
  }

  void PrintModelInfos() const {
    if (!config_.debug) {
      return;
    }
    PrintModelInfo(dp_sess_.get(), "duration_predictor");
    PrintModelInfo(text_enc_sess_.get(), "text_encoder");
    PrintModelInfo(vector_est_sess_.get(), "vector_estimator");
    PrintModelInfo(vocoder_sess_.get(), "vocoder");
  }

  void InitDurationPredictor(void *model_data, size_t model_data_length) {
    if (model_data) {
      dp_sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                                model_data_length, sess_opts_);
    } else if (!dp_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize duration predictor session "
          "outside of this function");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(dp_sess_.get(), &dp_input_names_, &dp_input_names_ptr_);
    GetOutputNames(dp_sess_.get(), &dp_output_names_, &dp_output_names_ptr_);
  }

  void InitTextEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      text_enc_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!text_enc_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize text encoder session outside "
          "of this function");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(text_enc_sess_.get(), &text_enc_input_names_,
                  &text_enc_input_names_ptr_);
    GetOutputNames(text_enc_sess_.get(), &text_enc_output_names_,
                   &text_enc_output_names_ptr_);
  }

  void InitVectorEstimator(void *model_data, size_t model_data_length) {
    if (model_data) {
      vector_est_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!vector_est_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize vector estimator session "
          "outside of this function");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(vector_est_sess_.get(), &vector_est_input_names_,
                  &vector_est_input_names_ptr_);
    GetOutputNames(vector_est_sess_.get(), &vector_est_output_names_,
                   &vector_est_output_names_ptr_);
  }

  void InitVocoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      vocoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!vocoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize vocoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }
    GetInputNames(vocoder_sess_.get(), &vocoder_input_names_,
                  &vocoder_input_names_ptr_);
    GetOutputNames(vocoder_sess_.get(), &vocoder_output_names_,
                   &vocoder_output_names_ptr_);
  }

  void LoadModels() {
    dp_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.supertonic.duration_predictor),
        sess_opts_);
    InitDurationPredictor(nullptr, 0);

    text_enc_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.supertonic.text_encoder),
        sess_opts_);
    InitTextEncoder(nullptr, 0);

    vector_est_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.supertonic.vector_estimator),
        sess_opts_);
    InitVectorEstimator(nullptr, 0);

    vocoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.supertonic.vocoder), sess_opts_);
    InitVocoder(nullptr, 0);
  }

  template <typename Manager>
  void LoadOneModel(Manager *mgr, const std::string &path,
                    const char *model_name,
                    const std::function<void(void *, size_t)> &init) {
    auto buf = ReadFile(mgr, path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read %s model: %s", model_name, path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    init(buf.data(), buf.size());
  }

  template <typename Manager>
  void LoadModels(Manager *mgr) {
    LoadOneModel(
        mgr, config_.supertonic.duration_predictor, "duration_predictor",
        [this](void *p, size_t len) { InitDurationPredictor(p, len); });
    LoadOneModel(mgr, config_.supertonic.text_encoder, "text_encoder",
                 [this](void *p, size_t len) { InitTextEncoder(p, len); });
    LoadOneModel(mgr, config_.supertonic.vector_estimator, "vector_estimator",
                 [this](void *p, size_t len) { InitVectorEstimator(p, len); });
    LoadOneModel(mgr, config_.supertonic.vocoder, "vocoder",
                 [this](void *p, size_t len) { InitVocoder(p, len); });
  }

  void Init() {
    std::string tts_config_path =
        ResolveAbsolutePath(config_.supertonic.tts_config);
    LoadConfig(tts_config_path);
    PrintDebugInfo(tts_config_path);
    LoadModels();
    PrintModelInfos();
  }

  template <typename Manager>
  void Init(Manager *mgr) {
    std::string tts_config_path =
        ResolveAbsolutePath(config_.supertonic.tts_config);
    LoadConfig(mgr, tts_config_path);
    PrintDebugInfo(tts_config_path);
    LoadModels(mgr);
    PrintModelInfos();
  }

  // Load config from binary (4 x int32 LE: sample_rate, base_chunk_size,
  // chunk_compress_factor, latent_dim). Shared by both LoadConfig variants.
  void LoadConfigFromBinary(const char *data, size_t size) {
    const size_t kExpectedSize = 4 * sizeof(int32_t);
    if (size < kExpectedSize) {
      SHERPA_ONNX_LOGE("tts.bin too small: %zu (expected %zu)", size,
                       kExpectedSize);
      SHERPA_ONNX_EXIT(-1);
    }
    int32_t vals[4];
    std::memcpy(vals, data, kExpectedSize);
    cfg_.ae.sample_rate = vals[0];
    cfg_.ae.base_chunk_size = vals[1];
    cfg_.ttl.chunk_compress_factor = vals[2];
    cfg_.ttl.latent_dim = vals[3];
    if (cfg_.ae.sample_rate <= 0) {
      SHERPA_ONNX_LOGE("Invalid sample_rate: %d", cfg_.ae.sample_rate);
      SHERPA_ONNX_EXIT(-1);
    }
    if (cfg_.ae.base_chunk_size <= 0) {
      SHERPA_ONNX_LOGE("Invalid base_chunk_size: %d", cfg_.ae.base_chunk_size);
      SHERPA_ONNX_EXIT(-1);
    }
    if (cfg_.ttl.chunk_compress_factor <= 0) {
      SHERPA_ONNX_LOGE("Invalid chunk_compress_factor: %d",
                       cfg_.ttl.chunk_compress_factor);
      SHERPA_ONNX_EXIT(-1);
    }
    if (cfg_.ttl.latent_dim <= 0) {
      SHERPA_ONNX_LOGE("Invalid latent_dim: %d", cfg_.ttl.latent_dim);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void LoadConfig(const std::string &config_path) {
    std::vector<char> buf = ReadFile(config_path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read config: %s", config_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    LoadConfigFromBinary(buf.data(), buf.size());
  }

  template <typename Manager>
  void LoadConfig(Manager *mgr, const std::string &config_path) {
    std::vector<char> buf = ReadFile(mgr, config_path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read config: %s", config_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    LoadConfigFromBinary(buf.data(), buf.size());
  }

  void InitCudaIOBinding() {
    use_cuda_iobinding_ =
        (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
  }

  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  SupertonicConfig cfg_;
  Ort::MemoryInfo cpu_mem_info_;
  std::unique_ptr<Ort::MemoryInfo> cuda_mem_info_;
  bool use_cuda_iobinding_ = false;
  bool is_cpu_provider_ = false;

  std::unique_ptr<Ort::Session> dp_sess_;
  std::unique_ptr<Ort::Session> text_enc_sess_;
  std::unique_ptr<Ort::Session> vector_est_sess_;
  std::unique_ptr<Ort::Session> vocoder_sess_;

  std::vector<std::string> dp_input_names_;
  std::vector<const char *> dp_input_names_ptr_;
  std::vector<std::string> dp_output_names_;
  std::vector<const char *> dp_output_names_ptr_;

  std::vector<std::string> text_enc_input_names_;
  std::vector<const char *> text_enc_input_names_ptr_;
  std::vector<std::string> text_enc_output_names_;
  std::vector<const char *> text_enc_output_names_ptr_;

  std::vector<std::string> vector_est_input_names_;
  std::vector<const char *> vector_est_input_names_ptr_;
  std::vector<std::string> vector_est_output_names_;
  std::vector<const char *> vector_est_output_names_ptr_;

  std::vector<std::string> vocoder_input_names_;
  std::vector<const char *> vocoder_input_names_ptr_;
  std::vector<std::string> vocoder_output_names_;
  std::vector<const char *> vocoder_output_names_ptr_;
};

const SupertonicConfig &OfflineTtsSupertonicModel::GetConfig() const {
  return impl_->GetConfig();
}

int32_t OfflineTtsSupertonicModel::GetSampleRate() const {
  return impl_->GetSampleRate();
}

Ort::Value OfflineTtsSupertonicModel::RunDurationPredictor(
    std::vector<Ort::Value> inputs) const {
  return impl_->RunDurationPredictor(std::move(inputs));
}

Ort::Value OfflineTtsSupertonicModel::RunTextEncoder(
    std::vector<Ort::Value> inputs) const {
  return impl_->RunTextEncoder(std::move(inputs));
}

Ort::Value OfflineTtsSupertonicModel::RunVectorEstimator(
    std::vector<Ort::Value> inputs) const {
  return impl_->RunVectorEstimator(std::move(inputs));
}

Ort::Value OfflineTtsSupertonicModel::RunVocoder(
    std::vector<Ort::Value> inputs) const {
  return impl_->RunVocoder(std::move(inputs));
}

bool OfflineTtsSupertonicModel::UseCudaIOBinding() const {
  return impl_->UseCudaIOBinding();
}

const Ort::MemoryInfo &OfflineTtsSupertonicModel::GetCpuMemoryInfo() const {
  return impl_->GetCpuMemoryInfo();
}

const Ort::MemoryInfo *OfflineTtsSupertonicModel::GetCudaMemoryInfo() const {
  return impl_->GetCudaMemoryInfo();
}

std::string OfflineTtsSupertonicModel::GetProvider() const {
  return impl_->GetProvider();
}

OfflineTtsSupertonicModel::OfflineTtsSupertonicModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsSupertonicModel::OfflineTtsSupertonicModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsSupertonicModel::~OfflineTtsSupertonicModel() = default;

#if __ANDROID_API__ >= 9
template OfflineTtsSupertonicModel::OfflineTtsSupertonicModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsSupertonicModel::OfflineTtsSupertonicModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
