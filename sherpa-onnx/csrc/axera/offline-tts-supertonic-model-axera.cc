// sherpa-onnx/csrc/axera/offline-tts-supertonic-model-axera.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axera/offline-tts-supertonic-model-axera.h"

#include <algorithm>
#include <array>
#include <cstring>
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

#include "ax_engine_api.h"  // NOLINT
#include "ax_sys_api.h"     // NOLINT
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

using json = nlohmann::json;

namespace {

struct SubModel {
  ~SubModel() {
    FreeIO(&io_data);
    if (handle) {
      AX_ENGINE_DestroyHandle(handle);
    }
  }

  AX_ENGINE_HANDLE handle = nullptr;
  AX_ENGINE_IO_INFO_T *io_info = nullptr;
  AX_ENGINE_IO_T io_data;
  std::mutex mutex;
};

void InitSubModel(const std::vector<char> &buf, SubModel *model, bool debug) {
  if (buf.empty()) {
    SHERPA_ONNX_LOGE("Empty model buffer");
    SHERPA_ONNX_EXIT(-1);
  }
  InitContext(buf.data(), buf.size(), debug, &model->handle);
  InitInputOutputAttrs(model->handle, debug, &model->io_info);
  PrepareIO(model->io_info, &model->io_data, debug);
}

}  // namespace

class OfflineTtsSupertonicModelAxera::Impl {
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
    return RunModel(dp_.get(), "duration_predictor",
                    {reinterpret_cast<const char *>(text_ids_i32.data()),
                     reinterpret_cast<const char *>(style_dp.data()),
                     reinterpret_cast<const char *>(text_mask.data())},
                    {text_ids_i32.size() * sizeof(int32_t),
                     style_dp.size() * sizeof(float),
                     text_mask.size() * sizeof(float)});
  }

  std::vector<float> RunTextEncoder(const std::vector<int64_t> &text_ids,
                                    const std::vector<float> &style_ttl,
                                    const std::vector<float> &text_mask) const {
    std::vector<int32_t> text_ids_i32(text_ids.begin(), text_ids.end());
    return RunModel(text_enc_.get(), "text_encoder",
                    {reinterpret_cast<const char *>(text_ids_i32.data()),
                     reinterpret_cast<const char *>(style_ttl.data()),
                     reinterpret_cast<const char *>(text_mask.data())},
                    {text_ids_i32.size() * sizeof(int32_t),
                     style_ttl.size() * sizeof(float),
                     text_mask.size() * sizeof(float)});
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
        {reinterpret_cast<const char *>(noisy_latent.data()),
         reinterpret_cast<const char *>(text_emb.data()),
         reinterpret_cast<const char *>(style_ttl.data()),
         reinterpret_cast<const char *>(latent_mask.data()),
         reinterpret_cast<const char *>(text_mask.data()),
         reinterpret_cast<const char *>(current_step.data()),
         reinterpret_cast<const char *>(total_step.data())},
        {noisy_latent.size() * sizeof(float), text_emb.size() * sizeof(float),
         style_ttl.size() * sizeof(float), latent_mask.size() * sizeof(float),
         text_mask.size() * sizeof(float), current_step.size() * sizeof(float),
         total_step.size() * sizeof(float)});
  }

  std::vector<float> RunVocoder(const std::vector<float> &latent) const {
    return RunModel(vocoder_.get(), "vocoder",
                    {reinterpret_cast<const char *>(latent.data())},
                    {latent.size() * sizeof(float)});
  }

 private:
  std::vector<float> RunModel(SubModel *model, const char *name,
                              const std::vector<const char *> &inputs,
                              const std::vector<size_t> &input_sizes) const {
    std::lock_guard<std::mutex> lock(model->mutex);

    if (model->io_info->nInputSize != inputs.size()) {
      SHERPA_ONNX_LOGE(
          "%s: Input count mismatch. model expects %u inputs, but got %zu",
          name, model->io_info->nInputSize, inputs.size());
      SHERPA_ONNX_EXIT(-1);
    }

    for (AX_U32 i = 0; i < model->io_info->nInputSize; ++i) {
      size_t expected = model->io_info->pInputs[i].nSize;
      if (expected != input_sizes[i]) {
        SHERPA_ONNX_LOGE(
            "%s: Input %u size mismatch. model expects %u bytes, but got %zu "
            "bytes",
            name, i, expected, input_sizes[i]);
        SHERPA_ONNX_EXIT(-1);
      }
      std::memcpy(model->io_data.pInputs[i].pVirAddr, inputs[i],
                  input_sizes[i]);
    }

    auto ret = AX_ENGINE_RunSync(model->handle, &model->io_data);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed for %s, ret = %d", name, ret);
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = model->io_info->pOutputs[0];
    auto &out_buf = model->io_data.pOutputs[0];
    size_t out_elems = out_meta.nSize / sizeof(float);
    std::vector<float> out(out_elems);
    std::memcpy(out.data(), out_buf.pVirAddr, out_meta.nSize);
    return out;
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
      InitSubModel(buf, out->get(), config_.debug);
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Axera sub-model %s loaded.", name);
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
      InitSubModel(buf, out->get(), config_.debug);
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Axera sub-model %s loaded.", name);
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
  AxEngineGuard ax_engine_guard_;
  SupertonicConfig cfg_;

  std::unique_ptr<SubModel> dp_;
  std::unique_ptr<SubModel> text_enc_;
  std::unique_ptr<SubModel> vector_est_;
  std::unique_ptr<SubModel> vocoder_;
};

OfflineTtsSupertonicModelAxera::~OfflineTtsSupertonicModelAxera() =
    default;

OfflineTtsSupertonicModelAxera::OfflineTtsSupertonicModelAxera(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsSupertonicModelAxera::OfflineTtsSupertonicModelAxera(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

const SupertonicConfig &OfflineTtsSupertonicModelAxera::GetConfig() const {
  return impl_->GetConfig();
}

int32_t OfflineTtsSupertonicModelAxera::GetSampleRate() const {
  return impl_->GetSampleRate();
}

std::vector<float> OfflineTtsSupertonicModelAxera::RunDurationPredictor(
    const std::vector<int64_t> &text_ids, const std::vector<float> &style_dp,
    const std::vector<float> &text_mask) const {
  return impl_->RunDurationPredictor(text_ids, style_dp, text_mask);
}

std::vector<float> OfflineTtsSupertonicModelAxera::RunTextEncoder(
    const std::vector<int64_t> &text_ids, const std::vector<float> &style_ttl,
    const std::vector<float> &text_mask) const {
  return impl_->RunTextEncoder(text_ids, style_ttl, text_mask);
}

std::vector<float> OfflineTtsSupertonicModelAxera::RunVectorEstimator(
    const std::vector<float> &noisy_latent,
    const std::vector<float> &current_step,
    const std::vector<float> &text_emb, const std::vector<float> &style_ttl,
    const std::vector<float> &latent_mask, const std::vector<float> &text_mask,
    const std::vector<float> &total_step) const {
  return impl_->RunVectorEstimator(noisy_latent, current_step, text_emb,
                                   style_ttl, latent_mask, text_mask,
                                   total_step);
}

std::vector<float> OfflineTtsSupertonicModelAxera::RunVocoder(
    const std::vector<float> &latent) const {
  return impl_->RunVocoder(latent);
}

#if __ANDROID_API__ >= 9
template OfflineTtsSupertonicModelAxera::OfflineTtsSupertonicModelAxera(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsSupertonicModelAxera::OfflineTtsSupertonicModelAxera(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
