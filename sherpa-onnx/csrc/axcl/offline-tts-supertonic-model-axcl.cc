// sherpa-onnx/csrc/axcl/offline-tts-supertonic-model-axcl.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axcl/offline-tts-supertonic-model-axcl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/axcl/axcl-model.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

using json = nlohmann::json;

namespace {

struct SubModel {
  std::unique_ptr<AxclModel> model;
  mutable std::mutex mutex;
};

}  // namespace

class OfflineTtsSupertonicModelAxcl::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config) : config_(config) {
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config) {
    Init(mgr);
  }

  ~Impl() = default;

  const SupertonicConfig &GetConfig() const { return cfg_; }

  int32_t GetSampleRate() const { return cfg_.ae.sample_rate; }

  std::vector<float> RunDurationPredictor(
      const std::vector<int64_t> &text_ids, const std::vector<float> &style_dp,
      const std::vector<float> &text_mask) const {
    std::vector<int32_t> text_ids_i32(text_ids.begin(), text_ids.end());
    return RunModel(
        dp_.get(), "duration_predictor",
        {{"text_ids", text_ids_i32.data(), text_ids_i32.size(), true},
         {"style_dp", style_dp.data(), style_dp.size(), false},
         {"text_mask", text_mask.data(), text_mask.size(), false}});
  }

  std::vector<float> RunTextEncoder(const std::vector<int64_t> &text_ids,
                                    const std::vector<float> &style_ttl,
                                    const std::vector<float> &text_mask) const {
    std::vector<int32_t> text_ids_i32(text_ids.begin(), text_ids.end());
    return RunModel(
        text_enc_.get(), "text_encoder",
        {{"text_ids", text_ids_i32.data(), text_ids_i32.size(), true},
         {"style_ttl", style_ttl.data(), style_ttl.size(), false},
         {"text_mask", text_mask.data(), text_mask.size(), false}});
  }

  std::vector<float> RunVectorEstimator(
      const std::vector<float> &noisy_latent,
      const std::vector<float> &current_step,
      const std::vector<float> &text_emb,
      const std::vector<float> &style_ttl,
      const std::vector<float> &latent_mask,
      const std::vector<float> &text_mask,
      const std::vector<float> &total_step) const {
    return RunModel(
        vector_est_.get(), "vector_estimator",
        {{"noisy_latent", noisy_latent.data(), noisy_latent.size(), false},
         {"text_emb", text_emb.data(), text_emb.size(), false},
         {"style_ttl", style_ttl.data(), style_ttl.size(), false},
         {"latent_mask", latent_mask.data(), latent_mask.size(), false},
         {"text_mask", text_mask.data(), text_mask.size(), false},
         {"current_step", current_step.data(), current_step.size(), false},
         {"total_step", total_step.data(), total_step.size(), false}});
  }

  std::vector<float> RunVocoder(const std::vector<float> &latent) const {
    return RunModel(vocoder_.get(), "vocoder",
                    {{"latent", latent.data(), latent.size(), false}});
  }

 private:
  struct InputDesc {
    std::string name;
    const void *data;
    size_t num_elements;
    bool is_int32;
  };

  std::vector<float> RunModel(SubModel *sub, const char *name,
                              const std::vector<InputDesc> &inputs) const {
    std::lock_guard<std::mutex> lock(sub->mutex);
    AxclModel *model = sub->model.get();

    if (model->InputTensorNames().size() != inputs.size()) {
      SHERPA_ONNX_LOGE(
          "%s: Input count mismatch. model expects %zu inputs, but got %zu",
          name, model->InputTensorNames().size(), inputs.size());
      SHERPA_ONNX_EXIT(-1);
    }

    for (const auto &in : inputs) {
      int32_t expected_bytes = model->TensorSizeInBytes(in.name);
      size_t actual_bytes = in.is_int32
                                ? in.num_elements * sizeof(int32_t)
                                : in.num_elements * sizeof(float);
      if (expected_bytes != static_cast<int32_t>(actual_bytes)) {
        SHERPA_ONNX_LOGE(
            "%s: Input '%s' size mismatch. model expects %d bytes, but got %zu "
            "bytes",
            name, in.name.c_str(), expected_bytes, actual_bytes);
        SHERPA_ONNX_EXIT(-1);
      }
      bool ok;
      if (in.is_int32) {
        ok = model->SetInputTensorData(
            in.name, reinterpret_cast<const int32_t *>(in.data),
            in.num_elements);
      } else {
        ok = model->SetInputTensorData(
            in.name, reinterpret_cast<const float *>(in.data),
            in.num_elements);
      }
      if (!ok) {
        SHERPA_ONNX_LOGE("%s: Failed to set input '%s'", name,
                         in.name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    if (!model->Run()) {
      SHERPA_ONNX_LOGE("Failed to run %s", name);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_names = model->OutputTensorNames();
    if (out_names.empty()) {
      SHERPA_ONNX_LOGE("%s: Model has no outputs", name);
      SHERPA_ONNX_EXIT(-1);
    }
    return model->GetOutputTensorData(out_names[0]);
  }

  void ParseConfig(const json &j) {
    if (j.find("ae") == j.end() || j.find("ttl") == j.end()) {
      SHERPA_ONNX_LOGE("Invalid config file: missing 'ae' or 'ttl' section");
      SHERPA_ONNX_EXIT(-1);
    }
    const auto &ae = j["ae"];
    const auto &ttl = j["ttl"];
    auto get_int = [](const json &obj, const char *key,
                      const char *section) -> int32_t {
      if (obj.find(key) == obj.end()) {
        SHERPA_ONNX_LOGE("Invalid config: %s.%s missing", section, key);
        SHERPA_ONNX_EXIT(-1);
      }
      if (!obj[key].is_number_integer()) {
        SHERPA_ONNX_LOGE("Invalid config: %s.%s must be integer", section, key);
        SHERPA_ONNX_EXIT(-1);
      }
      return obj[key].get<int32_t>();
    };
    cfg_.ae.sample_rate = get_int(ae, "sample_rate", "ae");
    cfg_.ae.base_chunk_size = get_int(ae, "base_chunk_size", "ae");
    cfg_.ttl.chunk_compress_factor =
        get_int(ttl, "chunk_compress_factor", "ttl");
    cfg_.ttl.latent_dim = get_int(ttl, "latent_dim", "ttl");
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

  static json LoadJsonFromBuffer(const std::vector<char> &buf) {
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Empty json buffer");
      SHERPA_ONNX_EXIT(-1);
    }
    try {
      return json::parse(buf.begin(), buf.end());
    } catch (const std::exception &e) {
      SHERPA_ONNX_LOGE("Failed to parse JSON buffer: %s", e.what());
      SHERPA_ONNX_EXIT(-1);
    }
    return json{};
  }

  void LoadConfig(const std::string &config_path) {
    auto buf = ReadFile(config_path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read config: %s", config_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    json j = LoadJsonFromBuffer(buf);
    ParseConfig(j);
  }

  template <typename Manager>
  void LoadConfig(Manager *mgr, const std::string &config_path) {
    auto buf = ReadFile(mgr, config_path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read config: %s", config_path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    json j = LoadJsonFromBuffer(buf);
    ParseConfig(j);
  }

  void LoadModels() {
    auto load = [this](const std::string &path, const char *name,
                       std::unique_ptr<SubModel> *out) {
      auto buf = ReadFile(path);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read %s model: %s", name, path.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      out->reset(new SubModel);
      out->get()->model = std::make_unique<AxclModel>(buf.data(), buf.size());
      if (!out->get()->model->IsInitialized()) {
        SHERPA_ONNX_LOGE("Failed to initialize %s model", name);
        SHERPA_ONNX_EXIT(-1);
      }
      if (config_.debug) {
        SHERPA_ONNX_LOGE("AXCL sub-model %s loaded.", name);
      }
    };

    load(config_.supertonic.duration_predictor, "duration_predictor",
         &dp_);
    load(config_.supertonic.text_encoder, "text_encoder", &text_enc_);
    load(config_.supertonic.vector_estimator, "vector_estimator",
         &vector_est_);
    load(config_.supertonic.vocoder, "vocoder", &vocoder_);
  }

  template <typename Manager>
  void LoadModels(Manager *mgr) {
    auto load = [this, mgr](const std::string &path, const char *name,
                            std::unique_ptr<SubModel> *out) {
      auto buf = ReadFile(mgr, path);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read %s model: %s", name, path.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      out->reset(new SubModel);
      out->get()->model = std::make_unique<AxclModel>(buf.data(), buf.size());
      if (!out->get()->model->IsInitialized()) {
        SHERPA_ONNX_LOGE("Failed to initialize %s model", name);
        SHERPA_ONNX_EXIT(-1);
      }
      if (config_.debug) {
        SHERPA_ONNX_LOGE("AXCL sub-model %s loaded.", name);
      }
    };

    load(config_.supertonic.duration_predictor, "duration_predictor",
         &dp_);
    load(config_.supertonic.text_encoder, "text_encoder", &text_enc_);
    load(config_.supertonic.vector_estimator, "vector_estimator",
         &vector_est_);
    load(config_.supertonic.vocoder, "vocoder", &vocoder_);
  }

  void Init() {
    std::string tts_config_path =
        ResolveAbsolutePath(config_.supertonic.tts_json);
    LoadConfig(tts_config_path);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Supertonic config: sample_rate=%d, base_chunk_size=%d, "
                       "chunk_compress_factor=%d, latent_dim=%d",
                       cfg_.ae.sample_rate, cfg_.ae.base_chunk_size,
                       cfg_.ttl.chunk_compress_factor, cfg_.ttl.latent_dim);
    }
    LoadModels();
  }

  template <typename Manager>
  void Init(Manager *mgr) {
    std::string tts_config_path =
        ResolveAbsolutePath(config_.supertonic.tts_json);
    LoadConfig(mgr, tts_config_path);
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Supertonic config: sample_rate=%d, base_chunk_size=%d, "
                       "chunk_compress_factor=%d, latent_dim=%d",
                       cfg_.ae.sample_rate, cfg_.ae.base_chunk_size,
                       cfg_.ttl.chunk_compress_factor, cfg_.ttl.latent_dim);
    }
    LoadModels(mgr);
  }

 private:
  OfflineTtsModelConfig config_;
  SupertonicConfig cfg_;

  std::unique_ptr<SubModel> dp_;
  std::unique_ptr<SubModel> text_enc_;
  std::unique_ptr<SubModel> vector_est_;
  std::unique_ptr<SubModel> vocoder_;
};

OfflineTtsSupertonicModelAxcl::~OfflineTtsSupertonicModelAxcl() = default;

OfflineTtsSupertonicModelAxcl::OfflineTtsSupertonicModelAxcl(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsSupertonicModelAxcl::OfflineTtsSupertonicModelAxcl(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

const SupertonicConfig &OfflineTtsSupertonicModelAxcl::GetConfig() const {
  return impl_->GetConfig();
}

int32_t OfflineTtsSupertonicModelAxcl::GetSampleRate() const {
  return impl_->GetSampleRate();
}

std::vector<float> OfflineTtsSupertonicModelAxcl::RunDurationPredictor(
    const std::vector<int64_t> &text_ids, const std::vector<float> &style_dp,
    const std::vector<float> &text_mask) const {
  return impl_->RunDurationPredictor(text_ids, style_dp, text_mask);
}

std::vector<float> OfflineTtsSupertonicModelAxcl::RunTextEncoder(
    const std::vector<int64_t> &text_ids, const std::vector<float> &style_ttl,
    const std::vector<float> &text_mask) const {
  return impl_->RunTextEncoder(text_ids, style_ttl, text_mask);
}

std::vector<float> OfflineTtsSupertonicModelAxcl::RunVectorEstimator(
    const std::vector<float> &noisy_latent,
    const std::vector<float> &current_step,
    const std::vector<float> &text_emb, const std::vector<float> &style_ttl,
    const std::vector<float> &latent_mask, const std::vector<float> &text_mask,
    const std::vector<float> &total_step) const {
  return impl_->RunVectorEstimator(noisy_latent, current_step, text_emb,
                                   style_ttl, latent_mask, text_mask,
                                   total_step);
}

std::vector<float> OfflineTtsSupertonicModelAxcl::RunVocoder(
    const std::vector<float> &latent) const {
  return impl_->RunVocoder(latent);
}

#if __ANDROID_API__ >= 9
template OfflineTtsSupertonicModelAxcl::OfflineTtsSupertonicModelAxcl(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsSupertonicModelAxcl::OfflineTtsSupertonicModelAxcl(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
