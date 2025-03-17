// sherpa-onnx/csrc/hifigan-vocoder.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/hifigan-vocoder.h"

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

namespace sherpa_onnx {

class HifiganVocoder::Impl {
 public:
  explicit Impl(int32_t num_threads, const std::string &provider,
                const std::string &model)
      : env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(num_threads, provider)),
        allocator_{} {
    auto buf = ReadFile(model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  explicit Impl(Manager *mgr, int32_t num_threads, const std::string &provider,
                const std::string &model)
      : env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(num_threads, provider)),
        allocator_{} {
    auto buf = ReadFile(mgr, model);
    Init(buf.data(), buf.size());
  }

  std::vector<float> Run(Ort::Value mel) const {
    auto out = sess_->Run({}, input_names_ptr_.data(), &mel, 1,
                          output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<int64_t> audio_shape =
        out[0].GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    // The output shape may be (1, 1, total) or (1, total) or (total,)
    for (auto i : audio_shape) {
      total *= i;
    }

    const float *p = out[0].GetTensorData<float>();
    return {p, p + total};
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
  }

 private:
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

HifiganVocoder::HifiganVocoder(int32_t num_threads, const std::string &provider,
                               const std::string &model)
    : impl_(std::make_unique<Impl>(num_threads, provider, model)) {}

template <typename Manager>
HifiganVocoder::HifiganVocoder(Manager *mgr, int32_t num_threads,
                               const std::string &provider,
                               const std::string &model)
    : impl_(std::make_unique<Impl>(mgr, num_threads, provider, model)) {}

HifiganVocoder::~HifiganVocoder() = default;

std::vector<float> HifiganVocoder::Run(Ort::Value mel) const {
  return impl_->Run(std::move(mel));
}

#if __ANDROID_API__ >= 9
template HifiganVocoder::HifiganVocoder(AAssetManager *mgr, int32_t num_threads,
                                        const std::string &provider,
                                        const std::string &model);
#endif

#if __OHOS__
template HifiganVocoder::HifiganVocoder(NativeResourceManager *mgr,
                                        int32_t num_threads,
                                        const std::string &provider,
                                        const std::string &model);
#endif

}  // namespace sherpa_onnx
