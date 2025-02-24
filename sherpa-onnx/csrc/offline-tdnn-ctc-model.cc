// sherpa-onnx/csrc/offline-tdnn-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tdnn-ctc-model.h"

#include <utility>

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
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineTdnnCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.tdnn.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.tdnn.model);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features) {
    auto nnet_out =
        sess_->Run({}, input_names_ptr_.data(), &features, 1,
                   output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<int64_t> nnet_out_shape =
        nnet_out[0].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<int64_t> out_length_vec(nnet_out_shape[0], nnet_out_shape[1]);
    std::vector<int64_t> out_length_shape(1, nnet_out_shape[0]);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value nnet_out_length = Ort::Value::CreateTensor(
        memory_info, out_length_vec.data(), out_length_vec.size(),
        out_length_shape.data(), out_length_shape.size());

    std::vector<Ort::Value> ans;
    ans.reserve(2);
    ans.push_back(std::move(nnet_out[0]));
    ans.push_back(Clone(Allocator(), &nnet_out_length));
    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

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
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
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
};

OfflineTdnnCtcModel::OfflineTdnnCtcModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTdnnCtcModel::OfflineTdnnCtcModel(Manager *mgr,
                                         const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTdnnCtcModel::~OfflineTdnnCtcModel() = default;

std::vector<Ort::Value> OfflineTdnnCtcModel::Forward(
    Ort::Value features, Ort::Value /*features_length*/) {
  return impl_->Forward(std::move(features));
}

int32_t OfflineTdnnCtcModel::VocabSize() const { return impl_->VocabSize(); }

OrtAllocator *OfflineTdnnCtcModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineTdnnCtcModel::OfflineTdnnCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineTdnnCtcModel::OfflineTdnnCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
