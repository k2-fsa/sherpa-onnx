// sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.cc
//
// Copyright (c)  2025  M5Stack Technology CO LTD

#include "sherpa-onnx/csrc/axcl/offline-sense-voice-model-axcl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/axcl/ax_model_runner_axcl.hpp"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OfflineSenseVoiceModelAxcl::Impl {
 public:
  ~Impl() { runner_.release(); }

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

    // input 0: features
    auto &in0 = runner_.get_input(0);
    size_t bytes0 = in0.nSize;
    if (bytes0 != features.size() * sizeof(float)) {
      SHERPA_ONNX_LOGE(
          "Feature size mismatch. model expects %u bytes, but got %zu bytes",
          in0.nSize, features.size() * sizeof(float));
      SHERPA_ONNX_EXIT(-1);
    }
    std::memcpy(in0.pVirAddr, features.data(), bytes0);

    auto &in1 = runner_.get_input(1);
    size_t bytes1 = in1.nSize;
    if (bytes1 != prompt.size() * sizeof(int32_t)) {
      SHERPA_ONNX_LOGE(
          "Prompt size mismatch. model expects %u bytes, but got %zu bytes",
          in1.nSize, prompt.size() * sizeof(int32_t));
      SHERPA_ONNX_EXIT(-1);
    }
    std::memcpy(in1.pVirAddr, prompt.data(), bytes1);

    int ret = runner_.inference();
    if (ret != 0) {
      SHERPA_ONNX_LOGE("ax_runner_axcl inference failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    // output 0
    auto &out0 = runner_.get_output(0);
    size_t out_elems = out0.nSize / sizeof(float);
    std::vector<float> out(out_elems);
    std::memcpy(out.data(), out0.pVirAddr, out0.nSize);
    return out;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    {
      if (auto ret = axclInit(0); 0 != ret) {
        fprintf(stderr, "Init AXCL failed{0x%8x}.\n", ret);
        return;
      }
      axclrtDeviceList lst;
      if (const auto ret = axclrtGetDeviceList(&lst);
          0 != ret || 0 == lst.num) {
        fprintf(stderr,
                "Get AXCL device failed{0x%8x}, find total %d device.\n", ret,
                lst.num);
        return;
      }
      if (const auto ret = axclrtSetDevice(lst.devices[0]); 0 != ret) {
        fprintf(stderr, "Set AXCL device failed{0x%8x}.\n", ret);
        return;
      }
      int ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
      if (0 != ret) {
        fprintf(stderr, "axclrtEngineInit %d\n", ret);
        return;
      }
    }

    int ret =
        runner_.init(reinterpret_cast<char *>(model_data), model_data_length);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("Init ax_runner_axcl failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    auto &in0 = runner_.get_input(0);
    if (in0.vShape.size() < 2) {
      SHERPA_ONNX_LOGE(
          "Input tensor rank is too small (rank = %zu). Shape vector is empty "
          "or has only 1 dim.",
          in0.vShape.size());
      SHERPA_ONNX_EXIT(-1);
    }
    num_input_frames_ = in0.vShape[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Axcl SenseVoice model init done with ax_runner_axcl.");
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
  ax_runner_axcl runner_;
  OfflineSenseVoiceModelMetaData meta_data_;
  int32_t num_input_frames_ = -1;
};

OfflineSenseVoiceModelAxcl::~OfflineSenseVoiceModelAxcl() = default;

OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineSenseVoiceModelAxcl::Run(std::vector<float> features,
                                                   int32_t language,
                                                   int32_t text_norm) const {
  return impl_->Run(std::move(features), language, text_norm);
}

const OfflineSenseVoiceModelMetaData &
OfflineSenseVoiceModelAxcl::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

#if __ANDROID_API__ >= 9
template OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineSenseVoiceModelAxcl::OfflineSenseVoiceModelAxcl(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx