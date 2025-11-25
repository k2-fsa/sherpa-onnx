// sherpa-onnx/csrc/axera/offline-sense-voice-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/offline-sense-voice-model-axera.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/axera/io.hpp"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAxera::Impl {
 public:
  ~Impl() {
    middleware::free_io(&io_data_);
    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  explicit Impl(const OfflineModelConfig &config) : config_(config) {
    auto buf = ReadFile(config_.sense_voice.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config) : config_(config) {
    auto buf = ReadFile(mgr, config_.sense_voice.model);
    Init(buf.data(), buf.size());
  }

  const OfflineSenseVoiceModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<float> features, int32_t language,
                         int32_t text_norm) {
    features = ApplyLFR(std::move(features));

    std::array<int32_t, 4> prompt{language, 1, 2, text_norm};

    if (!io_info_ || io_info_->nInputSize < 1) {
      SHERPA_ONNX_LOGE("Axera model expects at least 1 input tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &in0_meta = io_info_->pInputs[0];
    size_t bytes0 = in0_meta.nSize;

    if (bytes0 != features.size() * sizeof(float)) {
      SHERPA_ONNX_LOGE(
          "Feature size mismatch. model expects %u bytes, but got %zu bytes",
          in0_meta.nSize, features.size() * sizeof(float));
      SHERPA_ONNX_EXIT(-1);
    }

    std::memcpy(io_data_.pInputs[0].pVirAddr, features.data(), bytes0);

    //   io_info_->nInputSize >= 2
    //   io_info_->pInputs[1].nSize == prompt.size() * sizeof(int32_t)
    if (io_info_->nInputSize >= 2) {
      const auto &in1_meta = io_info_->pInputs[1];
      size_t bytes1 = in1_meta.nSize;
      if (bytes1 != prompt.size() * sizeof(int32_t)) {
        SHERPA_ONNX_LOGE(
            "Prompt size mismatch. model expects %u bytes, but got %zu bytes",
            in1_meta.nSize, prompt.size() * sizeof(int32_t));
        SHERPA_ONNX_EXIT(-1);
      }
      std::memcpy(io_data_.pInputs[1].pVirAddr, prompt.data(), bytes1);
    }

    auto ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    if (io_info_->nOutputSize < 1) {
      SHERPA_ONNX_LOGE("Axera model has no output tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_meta = io_info_->pOutputs[0];
    auto &out_buf = io_data_.pOutputs[0];

    size_t out_elems = out_meta.nSize / sizeof(float);
    std::vector<float> out(out_elems);

    std::memcpy(out.data(), out_buf.pVirAddr, out_meta.nSize);

    return out;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    InitEngine(config_.debug);

    InitContext(model_data, model_data_length, config_.debug, &handle_);

    InitInputOutputAttrs(handle_, config_.debug, &io_info_);

    std::memset(&io_data_, 0, sizeof(io_data_));

    PrepareIO(io_info_, &io_data_, config_.debug);

    if (!io_info_ || io_info_->nInputSize == 0 || !io_info_->pInputs) {
      SHERPA_ONNX_LOGE("No input tensor in Axera model");
      SHERPA_ONNX_EXIT(-1);
    }

    auto &in0 = io_info_->pInputs[0];
    if (in0.nShapeSize < 2) {
      SHERPA_ONNX_LOGE("Input tensor rank is too small (nShapeSize = %u)",
                       in0.nShapeSize);
      SHERPA_ONNX_EXIT(-1);
    }
    num_input_frames_ = in0.pShape[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Axera SenseVoice model init done.");
      SHERPA_ONNX_LOGE("  num_input_frames_ = %d", num_input_frames_);
    }
  }

  std::vector<float> ApplyLFR(std::vector<float> in) const {
    int32_t lfr_window_size = meta_data_.window_size;
    int32_t lfr_window_shift = meta_data_.window_shift;
    int32_t in_feat_dim = 80;
    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;

    if (out_num_frames > num_input_frames_) {
      SHERPA_ONNX_LOGE(
          "Number of input frames %d is too large. Truncate it to %d frames.",
          out_num_frames, num_input_frames_);
      SHERPA_ONNX_LOGE(
          "Recognition result may be truncated/incomplete. Please select a "
          "model accepting longer audios.");
      out_num_frames = num_input_frames_;
    }

    int32_t out_feat_dim = in_feat_dim * lfr_window_size;
    std::vector<float> out(num_input_frames_ * out_feat_dim);
    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);
      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

 private:
  OfflineModelConfig config_;
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_data_;
  OfflineSenseVoiceModelMetaData meta_data_;
  int32_t num_input_frames_ = -1;
};

OfflineSenseVoiceModelAxera::~OfflineSenseVoiceModelAxera() = default;

OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineSenseVoiceModelAxera::Run(std::vector<float> features,
                                                    int32_t language,
                                                    int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData &
OfflineSenseVoiceModelAxera::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModelAxera::OfflineSenseVoiceModelAxera(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx