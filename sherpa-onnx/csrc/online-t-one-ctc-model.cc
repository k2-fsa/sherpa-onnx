// sherpa-onnx/csrc/online-t-one-ctc-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-t-one-ctc-model.h"

#include <algorithm>
#include <cmath>
#include <string>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

class OnlineToneCtcModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.t_one_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.t_one_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  std::vector<Ort::Value> Forward(Ort::Value x,
                                  std::vector<Ort::Value> states) {
    // shape0 is (batch_size, 1, num_samples)
    auto shape0 = x.GetTensorTypeAndShapeInfo().GetShape();
    std::array<int64_t, 3> shape = {shape0[0], shape0[2], shape0[1]};
    std::vector<int32_t> samples(shape[0] * shape[1] * shape[2]);
    const float *px = x.GetTensorData<float>();

    for (int32_t i = 0; i < samples.size(); ++i) {
      float f = px[i];
      f = f > 1 ? 1 : f;
      f = f < -1 ? -1 : f;
      samples[i] = static_cast<int32_t>(f * 32767);
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value xx =
        Ort::Value::CreateTensor(memory_info, samples.data(), samples.size(),
                                 shape.data(), shape.size());

    std::array<Ort::Value, 2> inputs = {std::move(xx), std::move(states[0])};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    // out[0]: log_probs
    // out[1] next_states

    return out;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t ChunkLength() const { return 1; }

  int32_t ChunkShift() const { return 1; }

  OrtAllocator *Allocator() { return allocator_; }

  // Return a vector containing 1 tensor
  // - state_
  std::vector<Ort::Value> GetInitStates() {
    std::vector<Ort::Value> ans;
    ans.push_back(View(&state_));

    return ans;
  }

  std::vector<Ort::Value> StackStates(
      std::vector<std::vector<Ort::Value>> states) {
    int32_t batch_size = static_cast<int32_t>(states.size());
    if (batch_size == 1) {
      return std::move(states[0]);
    }

    std::vector<Ort::Value> ans;
    ans.reserve(1);

    std::vector<const Ort::Value *> buf(batch_size);

    for (int32_t b = 0; b != batch_size; ++b) {
      buf.push_back(&states[b][0]);
    }

    Ort::Value c{nullptr};
    c = Cat(allocator_, buf, 0);

    ans.push_back(std::move(c));

    return ans;
  }

  std::vector<std::vector<Ort::Value>> UnStackStates(
      std::vector<Ort::Value> states) const {
    auto allocator = const_cast<Impl *>(this)->allocator_;

    std::vector<std::vector<Ort::Value>> ans;

    auto shape = states[0].GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = shape[0];
    ans.resize(batch_size);

    if (batch_size == 1) {
      ans[0] = std::move(states);
      return ans;
    }

    std::vector<Ort::Value> v;
    v = Unbind(allocator, &states[0], 0);

    for (int32_t b = 0; b != batch_size; ++b) {
      ans[b].push_back(std::move(v[b]));
    }

    return ans;
  }

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
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(frame_length_ms_, "frame_length_ms");
    SHERPA_ONNX_READ_META_DATA(state_dim_, "state_dim");
    SHERPA_ONNX_READ_META_DATA(sample_rate_, "sample_rate");

    InitStates();

    vocab_size_ = sess_->GetOutputTypeInfo(0)
                      .GetTensorTypeAndShapeInfo()
                      .GetShape()
                      .back();
  }

  void InitStates() {
    std::array<int64_t, 2> state_shape{1, state_dim_};

    state_ = Ort::Value::CreateTensor(allocator_, state_shape.data(),
                                      state_shape.size(),
                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    auto p = state_.GetTensorMutableData<uint16_t>();
    std::fill(p, p + state_dim_, 0);
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  // One input frame is of  length is 300ms
  // For each input frame, there are 10 output frames,
  // so each output frame is 30ms
  int32_t frame_length_ms_ = 0;
  int32_t state_dim_ = 0;
  int32_t sample_rate_ = 0;
  int32_t vocab_size_ = 0;

  Ort::Value state_{nullptr};
};

OnlineToneCtcModel::OnlineToneCtcModel(const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineToneCtcModel::OnlineToneCtcModel(Manager *mgr,
                                       const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineToneCtcModel::~OnlineToneCtcModel() = default;

std::vector<Ort::Value> OnlineToneCtcModel::Forward(
    Ort::Value x, std::vector<Ort::Value> states) const {
  return impl_->Forward(std::move(x), std::move(states));
}

int32_t OnlineToneCtcModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OnlineToneCtcModel::ChunkLength() const { return impl_->ChunkLength(); }

int32_t OnlineToneCtcModel::ChunkShift() const { return impl_->ChunkShift(); }

OrtAllocator *OnlineToneCtcModel::Allocator() const {
  return impl_->Allocator();
}

std::vector<Ort::Value> OnlineToneCtcModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::vector<Ort::Value> OnlineToneCtcModel::StackStates(
    std::vector<std::vector<Ort::Value>> states) const {
  return impl_->StackStates(std::move(states));
}

std::vector<std::vector<Ort::Value>> OnlineToneCtcModel::UnStackStates(
    std::vector<Ort::Value> states) const {
  return impl_->UnStackStates(std::move(states));
}

#if __ANDROID_API__ >= 9
template OnlineToneCtcModel::OnlineToneCtcModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineToneCtcModel::OnlineToneCtcModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
