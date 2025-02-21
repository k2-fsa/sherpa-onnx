// sherpa-onnx/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"

#include <algorithm>
#include <cassert>
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
#include "sherpa-onnx/csrc/file-utils.h"

namespace sherpa_onnx {

class OnlineZipformerTransducerModelRknn::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }
  }

  std::vector<std::vector<uint8_t>> GetEncoderInitStates() const { return {}; }

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> RunEncoder(
      const std::vector<float> &features,
      std::vector<std::vector<uint8_t>> states) const {
    return {};
  }

  std::vector<float> RunDecoder(
      const std::vector<int64_t> &decoder_input) const {
    return {};
  }

  std::vector<float> RunJoiner(const std::vector<float> &encoder_out,
                               const std::vector<float> &decoder_out) const {
    return {};
  }

  int32_t ContextSize() const { return 0; }

  int32_t ChunkSize() const { return 0; }

  int32_t ChunkShift() const { return 0; }

  int32_t VocabSize() const { return 0; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    auto ret =
        rknn_init(&encoder_ctx_, model_data, model_data_length, 0, nullptr);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to init encoder '%s' with return code %d",
                       config_.transducer.encoder.c_str(), ret);
      SHERPA_ONNX_EXIT(0);
    }
    if (config_.debug) {
      rknn_sdk_version v;
      ret = rknn_query(encoder_ctx_, RKNN_QUERY_SDK_VERSION, &v, sizeof(v));
      SHERPA_ONNX_LOGE("sdk api version: %s, driver version: %s", v.api_version,
                       v.drv_version);
    }
  }

 private:
  OnlineModelConfig config_;
  rknn_context encoder_ctx_ = 0;
};

OnlineZipformerTransducerModelRknn::~OnlineZipformerTransducerModelRknn() =
    default;

OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<OnlineZipformerTransducerModelRknn>(mgr, config)) {
}

std::vector<std::vector<uint8_t>>
OnlineZipformerTransducerModelRknn::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>>
OnlineZipformerTransducerModelRknn::RunEncoder(
    const std::vector<float> &features,
    std::vector<std::vector<uint8_t>> states) const {
  return impl_->RunEncoder(features, std::move(states));
}

std::vector<float> OnlineZipformerTransducerModelRknn::RunDecoder(
    const std::vector<int64_t> &decoder_input) const {
  return impl_->RunDecoder(decoder_input);
}

std::vector<float> OnlineZipformerTransducerModelRknn::RunJoiner(
    const std::vector<float> &encoder_out,
    const std::vector<float> &decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineZipformerTransducerModelRknn::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelRknn::ChunkSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelRknn::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerTransducerModelRknn::VocabSize() const {
  return impl_->VocabSize();
}

#if __ANDROID_API__ >= 9
template OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
