// sherpa-onnx/csrc/offline-medasr-ctc-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-medasr-ctc-model.h"

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
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

std::vector<int64_t> GetMask(Ort::Value length) {
  auto shape = length.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 1) {
    SHERPA_ONNX_LOGE("Invalid length dim %zu", shape.size());
    SHERPA_ONNX_EXIT(-1);
  }

  auto batch_size = shape[0];

  const int64_t *p = length.GetTensorData<int64_t>();

  int64_t max_len = *std::max_element(p, p + batch_size);

  std::vector<int64_t> ans(batch_size * max_len, 0);

  int64_t *p_mask = ans.data();

  for (int32_t i = 0; i < batch_size; ++i) {
    auto len = p[i];
    std::fill(p_mask, p_mask + len, 1);

    p_mask += max_len;
  }

  return ans;
}

}  // namespace

class OfflineMedAsrCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.medasr.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.medasr.model);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    std::vector<int64_t> mask = GetMask(std::move(features_length));

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> shape =
        features.GetTensorTypeAndShapeInfo().GetShape();
    shape.resize(2);

    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, mask.data(), mask.size(), shape.data(), shape.size());

    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(mask_tensor)};

    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t SubsamplingFactor() const { return 4; }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
    if (model_type != "medasr_ctc") {
      SHERPA_ONNX_LOGE("Expect model type medasr_ctc. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(subsampling_factor_,
                                            "subsampling_factor", 4);
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 0;
};

OfflineMedAsrCtcModel::OfflineMedAsrCtcModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineMedAsrCtcModel::OfflineMedAsrCtcModel(Manager *mgr,
                                             const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineMedAsrCtcModel::~OfflineMedAsrCtcModel() = default;

std::vector<Ort::Value> OfflineMedAsrCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineMedAsrCtcModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineMedAsrCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineMedAsrCtcModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineMedAsrCtcModel::OfflineMedAsrCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineMedAsrCtcModel::OfflineMedAsrCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
