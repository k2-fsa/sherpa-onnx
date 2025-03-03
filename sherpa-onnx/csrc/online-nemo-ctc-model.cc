// sherpa-onnx/csrc/online-nemo-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-nemo-ctc-model.h"

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
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

class OnlineNeMoCtcModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.nemo_ctc.model);
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
      auto buf = ReadFile(mgr, config.nemo_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  std::vector<Ort::Value> Forward(Ort::Value x,
                                  std::vector<Ort::Value> states) {
    Ort::Value &cache_last_channel = states[0];
    Ort::Value &cache_last_time = states[1];
    Ort::Value &cache_last_channel_len = states[2];

    int32_t batch_size = x.GetTensorTypeAndShapeInfo().GetShape()[0];

    std::array<int64_t, 1> length_shape{batch_size};

    Ort::Value length = Ort::Value::CreateTensor<int64_t>(
        allocator_, length_shape.data(), length_shape.size());

    int64_t *p_length = length.GetTensorMutableData<int64_t>();

    std::fill(p_length, p_length + batch_size, ChunkLength());

    // (B, T, C) -> (B, C, T)
    x = Transpose12(allocator_, &x);

    std::array<Ort::Value, 5> inputs = {
        std::move(x), View(&length), std::move(cache_last_channel),
        std::move(cache_last_time), std::move(cache_last_channel_len)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    // out[0]: logit
    // out[1] logit_length
    // out[2:] states_next
    //
    // we need to remove out[1]

    std::vector<Ort::Value> ans;
    ans.reserve(out.size() - 1);

    for (int32_t i = 0; i != out.size(); ++i) {
      if (i == 1) {
        continue;
      }

      ans.push_back(std::move(out[i]));
    }

    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t ChunkLength() const { return window_size_; }

  int32_t ChunkShift() const { return chunk_shift_; }

  OrtAllocator *Allocator() { return allocator_; }

  // Return a vector containing 3 tensors
  // - cache_last_channel
  // - cache_last_time_
  // - cache_last_channel_len
  std::vector<Ort::Value> GetInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(3);
    ans.push_back(View(&cache_last_channel_));
    ans.push_back(View(&cache_last_time_));
    ans.push_back(View(&cache_last_channel_len_));

    return ans;
  }

  std::vector<Ort::Value> StackStates(
      std::vector<std::vector<Ort::Value>> states) {
    int32_t batch_size = static_cast<int32_t>(states.size());
    if (batch_size == 1) {
      return std::move(states[0]);
    }

    std::vector<Ort::Value> ans;

    // stack cache_last_channel
    std::vector<const Ort::Value *> buf(batch_size);

    // there are 3 states to be stacked
    for (int32_t i = 0; i != 3; ++i) {
      buf.clear();
      buf.reserve(batch_size);

      for (int32_t b = 0; b != batch_size; ++b) {
        assert(states[b].size() == 3);
        buf.push_back(&states[b][i]);
      }

      Ort::Value c{nullptr};
      if (i == 2) {
        c = Cat<int64_t>(allocator_, buf, 0);
      } else {
        c = Cat(allocator_, buf, 0);
      }

      ans.push_back(std::move(c));
    }

    return ans;
  }

  std::vector<std::vector<Ort::Value>> UnStackStates(
      std::vector<Ort::Value> states) const {
    assert(states.size() == 3);

    auto allocator = const_cast<Impl *>(this)->allocator_;

    std::vector<std::vector<Ort::Value>> ans;

    auto shape = states[0].GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = shape[0];
    ans.resize(batch_size);

    if (batch_size == 1) {
      ans[0] = std::move(states);
      return ans;
    }

    for (int32_t i = 0; i != 3; ++i) {
      std::vector<Ort::Value> v;
      if (i == 2) {
        v = Unbind<int64_t>(allocator, &states[i], 0);
      } else {
        v = Unbind(allocator, &states[i], 0);
      }

      assert(v.size() == batch_size);

      for (int32_t b = 0; b != batch_size; ++b) {
        ans[b].push_back(std::move(v[b]));
      }
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
    SHERPA_ONNX_READ_META_DATA(window_size_, "window_size");
    SHERPA_ONNX_READ_META_DATA(chunk_shift_, "chunk_shift");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim1_,
                               "cache_last_channel_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim2_,
                               "cache_last_channel_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_channel_dim3_,
                               "cache_last_channel_dim3");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim1_, "cache_last_time_dim1");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim2_, "cache_last_time_dim2");
    SHERPA_ONNX_READ_META_DATA(cache_last_time_dim3_, "cache_last_time_dim3");

    // need to increase by 1 since the blank token is not included in computing
    // vocab_size in NeMo.
    vocab_size_ += 1;

    InitStates();
  }

  void InitStates() {
    std::array<int64_t, 4> cache_last_channel_shape{1, cache_last_channel_dim1_,
                                                    cache_last_channel_dim2_,
                                                    cache_last_channel_dim3_};

    cache_last_channel_ = Ort::Value::CreateTensor<float>(
        allocator_, cache_last_channel_shape.data(),
        cache_last_channel_shape.size());

    Fill<float>(&cache_last_channel_, 0);

    std::array<int64_t, 4> cache_last_time_shape{
        1, cache_last_time_dim1_, cache_last_time_dim2_, cache_last_time_dim3_};

    cache_last_time_ = Ort::Value::CreateTensor<float>(
        allocator_, cache_last_time_shape.data(), cache_last_time_shape.size());

    Fill<float>(&cache_last_time_, 0);

    int64_t shape = 1;
    cache_last_channel_len_ =
        Ort::Value::CreateTensor<int64_t>(allocator_, &shape, 1);

    cache_last_channel_len_.GetTensorMutableData<int64_t>()[0] = 0;
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

  int32_t window_size_ = 0;
  int32_t chunk_shift_ = 0;
  int32_t subsampling_factor_ = 0;
  int32_t vocab_size_ = 0;
  int32_t cache_last_channel_dim1_ = 0;
  int32_t cache_last_channel_dim2_ = 0;
  int32_t cache_last_channel_dim3_ = 0;
  int32_t cache_last_time_dim1_ = 0;
  int32_t cache_last_time_dim2_ = 0;
  int32_t cache_last_time_dim3_ = 0;

  Ort::Value cache_last_channel_{nullptr};
  Ort::Value cache_last_time_{nullptr};
  Ort::Value cache_last_channel_len_{nullptr};
};

OnlineNeMoCtcModel::OnlineNeMoCtcModel(const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineNeMoCtcModel::OnlineNeMoCtcModel(Manager *mgr,
                                       const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineNeMoCtcModel::~OnlineNeMoCtcModel() = default;

std::vector<Ort::Value> OnlineNeMoCtcModel::Forward(
    Ort::Value x, std::vector<Ort::Value> states) const {
  return impl_->Forward(std::move(x), std::move(states));
}

int32_t OnlineNeMoCtcModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OnlineNeMoCtcModel::ChunkLength() const { return impl_->ChunkLength(); }

int32_t OnlineNeMoCtcModel::ChunkShift() const { return impl_->ChunkShift(); }

OrtAllocator *OnlineNeMoCtcModel::Allocator() const {
  return impl_->Allocator();
}

std::vector<Ort::Value> OnlineNeMoCtcModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::vector<Ort::Value> OnlineNeMoCtcModel::StackStates(
    std::vector<std::vector<Ort::Value>> states) const {
  return impl_->StackStates(std::move(states));
}

std::vector<std::vector<Ort::Value>> OnlineNeMoCtcModel::UnStackStates(
    std::vector<Ort::Value> states) const {
  return impl_->UnStackStates(std::move(states));
}

#if __ANDROID_API__ >= 9
template OnlineNeMoCtcModel::OnlineNeMoCtcModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineNeMoCtcModel::OnlineNeMoCtcModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
