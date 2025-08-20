// sherpa-onnx/csrc/online-zipformer2-ctc-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer2-ctc-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
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

class OnlineZipformer2CtcModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.zipformer2_ctc.model);
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
      auto buf = ReadFile(mgr, config.zipformer2_ctc.model);
      Init(buf.data(), buf.size());
    }
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  std::vector<Ort::Value> states) {
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + states.size());

    inputs.push_back(std::move(features));
    for (auto &v : states) {
      inputs.push_back(std::move(v));
    }

    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t ChunkLength() const { return T_; }

  int32_t ChunkShift() const { return decode_chunk_len_; }

  bool UseWhisperFeature() const { return use_whisper_feature_; }

  OrtAllocator *Allocator() { return allocator_; }

  // Return a vector containing 3 tensors
  // - attn_cache
  // - conv_cache
  // - offset
  std::vector<Ort::Value> GetInitStates() {
    std::vector<Ort::Value> ans;
    ans.reserve(initial_states_.size());
    for (auto &s : initial_states_) {
      ans.push_back(View(&s));
    }
    return ans;
  }

  std::vector<Ort::Value> StackStates(
      std::vector<std::vector<Ort::Value>> states) {
    int32_t batch_size = static_cast<int32_t>(states.size());

    std::vector<const Ort::Value *> buf(batch_size);

    std::vector<Ort::Value> ans;
    int32_t num_states = static_cast<int32_t>(states[0].size());
    ans.reserve(num_states);

    for (int32_t i = 0; i != (num_states - 2) / 6; ++i) {
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 1];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 2];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 3];
        }
        auto v = Cat(allocator_, buf, 1);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 4];
        }
        auto v = Cat(allocator_, buf, 0);
        ans.push_back(std::move(v));
      }
      {
        for (int32_t n = 0; n != batch_size; ++n) {
          buf[n] = &states[n][6 * i + 5];
        }
        auto v = Cat(allocator_, buf, 0);
        ans.push_back(std::move(v));
      }
    }

    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][num_states - 2];
      }
      auto v = Cat(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }

    {
      for (int32_t n = 0; n != batch_size; ++n) {
        buf[n] = &states[n][num_states - 1];
      }
      auto v = Cat<int64_t>(allocator_, buf, 0);
      ans.push_back(std::move(v));
    }
    return ans;
  }

  std::vector<std::vector<Ort::Value>> UnStackStates(
      std::vector<Ort::Value> states) {
    int32_t m = std::accumulate(num_encoder_layers_.begin(),
                                num_encoder_layers_.end(), 0);
    assert(states.size() == m * 6 + 2);

    int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];

    std::vector<std::vector<Ort::Value>> ans;
    ans.resize(batch_size);

    for (int32_t i = 0; i != m; ++i) {
      {
        auto v = Unbind(allocator_, &states[i * 6], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 1], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 2], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 3], 1);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 4], 0);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
      {
        auto v = Unbind(allocator_, &states[i * 6 + 5], 0);
        assert(v.size() == batch_size);

        for (int32_t n = 0; n != batch_size; ++n) {
          ans[n].push_back(std::move(v[n]));
        }
      }
    }

    {
      auto v = Unbind(allocator_, &states[m * 6], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
      }
    }
    {
      auto v = Unbind<int64_t>(allocator_, &states[m * 6 + 1], 0);
      assert(v.size() == batch_size);

      for (int32_t n = 0; n != batch_size; ++n) {
        ans[n].push_back(std::move(v[n]));
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
      os << "---zipformer2_ctc---\n";
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA_VEC(encoder_dims_, "encoder_dims");
    SHERPA_ONNX_READ_META_DATA_VEC(query_head_dims_, "query_head_dims");
    SHERPA_ONNX_READ_META_DATA_VEC(value_head_dims_, "value_head_dims");
    SHERPA_ONNX_READ_META_DATA_VEC(num_heads_, "num_heads");
    SHERPA_ONNX_READ_META_DATA_VEC(num_encoder_layers_, "num_encoder_layers");
    SHERPA_ONNX_READ_META_DATA_VEC(cnn_module_kernels_, "cnn_module_kernels");
    SHERPA_ONNX_READ_META_DATA_VEC(left_context_len_, "left_context_len");

    SHERPA_ONNX_READ_META_DATA(T_, "T");
    SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");

    std::string feature_type;
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(feature_type, "feature", "");
    if (feature_type == "whisper") {
      use_whisper_feature_ = true;
    }

    {
      auto shape =
          sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      vocab_size_ = shape[2];
    }

    if (config_.debug) {
      auto print = [](const std::vector<int32_t> &v, const char *name) {
        std::ostringstream os;
        os << name << ": ";
        for (auto i : v) {
          os << i << " ";
        }
        SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
      };
      print(encoder_dims_, "encoder_dims");
      print(query_head_dims_, "query_head_dims");
      print(value_head_dims_, "value_head_dims");
      print(num_heads_, "num_heads");
      print(num_encoder_layers_, "num_encoder_layers");
      print(cnn_module_kernels_, "cnn_module_kernels");
      print(left_context_len_, "left_context_len");
      SHERPA_ONNX_LOGE("T: %d", T_);
      SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);
      SHERPA_ONNX_LOGE("vocab_size_: %d", vocab_size_);
    }

    InitStates();
  }

  void InitStates() {
    int32_t n = static_cast<int32_t>(encoder_dims_.size());
    int32_t m = std::accumulate(num_encoder_layers_.begin(),
                                num_encoder_layers_.end(), 0);
    initial_states_.reserve(m * 6 + 2);

    for (int32_t i = 0; i != n; ++i) {
      int32_t num_layers = num_encoder_layers_[i];
      int32_t key_dim = query_head_dims_[i] * num_heads_[i];
      int32_t value_dim = value_head_dims_[i] * num_heads_[i];
      int32_t nonlin_attn_head_dim = 3 * encoder_dims_[i] / 4;

      for (int32_t j = 0; j != num_layers; ++j) {
        {
          std::array<int64_t, 3> s{left_context_len_[i], 1, key_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 4> s{1, 1, left_context_len_[i],
                                   nonlin_attn_head_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{left_context_len_[i], 1, value_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{left_context_len_[i], 1, value_dim};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{1, encoder_dims_[i],
                                   cnn_module_kernels_[i] / 2};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }

        {
          std::array<int64_t, 3> s{1, encoder_dims_[i],
                                   cnn_module_kernels_[i] / 2};
          auto v =
              Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
          Fill(&v, 0);
          initial_states_.push_back(std::move(v));
        }
      }
    }

    {
      std::array<int64_t, 4> s{1, 128, 3, 19};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      initial_states_.push_back(std::move(v));
    }

    {
      std::array<int64_t, 1> s{1};
      auto v =
          Ort::Value::CreateTensor<int64_t>(allocator_, s.data(), s.size());
      Fill<int64_t>(&v, 0);
      initial_states_.push_back(std::move(v));
    }
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

  std::vector<Ort::Value> initial_states_;

  std::vector<int32_t> encoder_dims_;
  std::vector<int32_t> query_head_dims_;
  std::vector<int32_t> value_head_dims_;
  std::vector<int32_t> num_heads_;
  std::vector<int32_t> num_encoder_layers_;
  std::vector<int32_t> cnn_module_kernels_;
  std::vector<int32_t> left_context_len_;

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;
  int32_t vocab_size_ = 0;

  // for models from
  // https://github.com/k2-fsa/icefall/blob/master/egs/multi_zh-hans/ASR/RESULTS.md#streaming-with-ctc-head
  bool use_whisper_feature_ = false;
};

OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OnlineZipformer2CtcModel::~OnlineZipformer2CtcModel() = default;

std::vector<Ort::Value> OnlineZipformer2CtcModel::Forward(
    Ort::Value x, std::vector<Ort::Value> states) const {
  return impl_->Forward(std::move(x), std::move(states));
}

int32_t OnlineZipformer2CtcModel::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineZipformer2CtcModel::ChunkLength() const {
  return impl_->ChunkLength();
}

int32_t OnlineZipformer2CtcModel::ChunkShift() const {
  return impl_->ChunkShift();
}

bool OnlineZipformer2CtcModel::UseWhisperFeature() const {
  return impl_->UseWhisperFeature();
}

OrtAllocator *OnlineZipformer2CtcModel::Allocator() const {
  return impl_->Allocator();
}

std::vector<Ort::Value> OnlineZipformer2CtcModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::vector<Ort::Value> OnlineZipformer2CtcModel::StackStates(
    std::vector<std::vector<Ort::Value>> states) const {
  return impl_->StackStates(std::move(states));
}

std::vector<std::vector<Ort::Value>> OnlineZipformer2CtcModel::UnStackStates(
    std::vector<Ort::Value> states) const {
  return impl_->UnStackStates(std::move(states));
}

#if __ANDROID_API__ >= 9
template OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformer2CtcModel::OnlineZipformer2CtcModel(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
