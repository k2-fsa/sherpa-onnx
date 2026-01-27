// sherpa-onnx/csrc/offline-tts-supertonic-model.cc
//
// Copyright (c)  2026 zengyw
//
// This file is based on Supertonic TTS
// (https://github.com/Supertone-Inc/supertonic) which is licensed under MIT
// License (Copyright (c) 2025 Supertone Inc.)

#include "sherpa-onnx/csrc/offline-tts-supertonic-model.h"

#include <fstream>
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

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

using json = nlohmann::json;

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

  Ort::Session *GetDurationPredictorSession() const { return dp_sess_.get(); }
  Ort::Session *GetTextEncoderSession() const { return text_enc_sess_.get(); }
  Ort::Session *GetVectorEstimatorSession() const {
    return vector_est_sess_.get();
  }
  Ort::Session *GetVocoderSession() const { return vocoder_sess_.get(); }

  bool UseCudaIOBinding() const { return use_cuda_iobinding_; }
  const Ort::MemoryInfo &GetCpuMemoryInfo() const { return cpu_mem_info_; }
  const Ort::MemoryInfo *GetCudaMemoryInfo() const {
    return cuda_mem_info_.get();
  }
  std::string GetProvider() const { return config_.provider; }

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

  void PrintDebugInfo(const std::string &model_dir) const {
    if (!config_.debug) {
      return;
    }
    std::ostringstream os;
    os << "---supertonic model---\n";
    os << "model_dir: " << model_dir << "\n";
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

  void LoadModels(const std::string &model_dir) {
    {
      auto buf = ReadFile(model_dir + "/duration_predictor.onnx");
      dp_sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                                sess_opts_);
    }
    {
      auto buf = ReadFile(model_dir + "/text_encoder.onnx");
      text_enc_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                      buf.size(), sess_opts_);
    }
    {
      auto buf = ReadFile(model_dir + "/vector_estimator.onnx");
      vector_est_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                        buf.size(), sess_opts_);
    }
    {
      auto buf = ReadFile(model_dir + "/vocoder.onnx");
      vocoder_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                     buf.size(), sess_opts_);
    }
  }

  template <typename Manager>
  void LoadModels(Manager *mgr, const std::string &model_dir) {
    {
      auto buf = ReadFile(mgr, model_dir + "/duration_predictor.onnx");
      dp_sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                                sess_opts_);
    }
    {
      auto buf = ReadFile(mgr, model_dir + "/text_encoder.onnx");
      text_enc_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                      buf.size(), sess_opts_);
    }
    {
      auto buf = ReadFile(mgr, model_dir + "/vector_estimator.onnx");
      vector_est_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                        buf.size(), sess_opts_);
    }
    {
      auto buf = ReadFile(mgr, model_dir + "/vocoder.onnx");
      vocoder_sess_ = std::make_unique<Ort::Session>(env_, buf.data(),
                                                     buf.size(), sess_opts_);
    }
  }

  void Init() {
    const std::string &model_dir =
        ResolveAbsolutePath(config_.supertonic.model_dir);
    LoadConfig(model_dir + "/tts.json");
    PrintDebugInfo(model_dir);
    LoadModels(model_dir);
    PrintModelInfos();
  }

  template <typename Manager>
  void Init(Manager *mgr) {
    const std::string &model_dir =
        ResolveAbsolutePath(config_.supertonic.model_dir);
    LoadConfig(model_dir + "/tts.json");
    PrintDebugInfo(model_dir);
    LoadModels(mgr, model_dir);
    PrintModelInfos();
  }

  void LoadConfig(const std::string &config_path) {
    AssertFileExists(config_path);
    std::ifstream file(config_path);
    if (!file.is_open()) {
      SHERPA_ONNX_LOGE("Failed to open config file: %s", config_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    json j;
    try {
      file >> j;
    } catch (const std::exception &e) {
      SHERPA_ONNX_LOGE("Failed to parse JSON config: %s", e.what());
      SHERPA_ONNX_EXIT(-1);
    }
    if (j.find("ae") == j.end() || j.find("ttl") == j.end()) {
      SHERPA_ONNX_LOGE("Invalid config file: missing 'ae' or 'ttl' section");
      SHERPA_ONNX_EXIT(-1);
    }
    cfg_.ae.sample_rate = j["ae"]["sample_rate"];
    cfg_.ae.base_chunk_size = j["ae"]["base_chunk_size"];
    cfg_.ttl.chunk_compress_factor = j["ttl"]["chunk_compress_factor"];
    cfg_.ttl.latent_dim = j["ttl"]["latent_dim"];
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
};

const SupertonicConfig &OfflineTtsSupertonicModel::GetConfig() const {
  return impl_->GetConfig();
}

int32_t OfflineTtsSupertonicModel::GetSampleRate() const {
  return impl_->GetSampleRate();
}

Ort::Session *OfflineTtsSupertonicModel::GetDurationPredictorSession() const {
  return impl_->GetDurationPredictorSession();
}

Ort::Session *OfflineTtsSupertonicModel::GetTextEncoderSession() const {
  return impl_->GetTextEncoderSession();
}

Ort::Session *OfflineTtsSupertonicModel::GetVectorEstimatorSession() const {
  return impl_->GetVectorEstimatorSession();
}

Ort::Session *OfflineTtsSupertonicModel::GetVocoderSession() const {
  return impl_->GetVocoderSession();
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
