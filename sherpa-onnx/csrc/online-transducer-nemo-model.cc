// sherpa-onnx/csrc/online-transducer-nemo-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

class OnlineTransducerNeMoModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }
  }
  
#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_WARNING),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.transducer.encoder_filename);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.decoder_filename);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.joiner_filename);
      InitJoiner(buf.data(), buf.size());
    }
  }
#endif

  std::vector<Ort::Value> StackStates(
      const std::vector<std::vector<Ort::Value>> &states) const {
    int32_t batch_size = static_cast<int32_t>(states.size());
    int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

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

  std::vector<std::vector<Ort::Value>>UnStackStates(
      const std::vector<Ort::Value> &states) const {
    int32_t m = std::accumulate(num_encoder_layers_.begin(),
                                num_encoder_layers_.end(), 0);
    assert(states.size() == m * 6 + 2);

    int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    int32_t num_encoders = num_encoder_layers_.size();

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

  std::pair<Ort::Value, std::vector<Ort::Value>>RunEncoder(Ort::Value features,
                                              std::vector<Ort::Value> states,
                                              Ort::Value /* processed_frames */) {
    std::vector<Ort::Value> encoder_inputs;
    encoder_inputs.reserve(1 + states.size());

    encoder_inputs.push_back(std::move(features));
    for (auto &v : states) {
      encoder_inputs.push_back(std::move(v));
    }

    auto encoder_out = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), encoder_inputs.data(),
        encoder_inputs.size(), encoder_output_names_ptr_.data(),
        encoder_output_names_ptr_.size());

    std::vector<Ort::Value> next_states;
    next_states.reserve(states.size());

    for (int32_t i = 1; i != static_cast<int32_t>(encoder_out.size()); ++i) {
      next_states.push_back(std::move(encoder_out[i]));
    }
    return {std::move(encoder_out[0]), std::move(next_states)};
  }

  Ort::Value RunDecoder(Ort::Value decoder_input) {
    auto decoder_out = decoder_sess_->Run(
        {}, decoder_input_names_ptr_.data(), &decoder_input, 1,
        decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
    return std::move(decoder_out[0]);
  }

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) {
    std::array<Ort::Value, 2> joiner_input = {std::move(encoder_out),
                                              std::move(decoder_out)};
    auto logit =
        joiner_sess_->Run({}, joiner_input_names_ptr_.data(), joiner_input.data(),
                          joiner_input.size(), joiner_output_names_ptr_.data(),
                          joiner_output_names_ptr_.size());

    return std::move(logit[0]);
}

  std::vector<Ort::Value> GetDecoderInitStates(int32_t batch_size) const {
    std::array<int64_t, 3> s0_shape{pred_rnn_layers_, batch_size, pred_hidden_};
    Ort::Value s0 = Ort::Value::CreateTensor<float>(allocator_, s0_shape.data(),
                                                    s0_shape.size());

    Fill<float>(&s0, 0);

    std::array<int64_t, 3> s1_shape{pred_rnn_layers_, batch_size, pred_hidden_};

    Ort::Value s1 = Ort::Value::CreateTensor<float>(allocator_, s1_shape.data(),
                                                    s1_shape.size());

    Fill<float>(&s1, 0);

    std::vector<Ort::Value> states;

    states.reserve(2);
    states.push_back(std::move(s0));
    states.push_back(std::move(s1));

    return states;
  }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t VocabSize() const { return vocab_size_; }

  OrtAllocator *Allocator() const { return allocator_; }

  std::string FeatureNormalizationMethod() const { return normalize_type_; }

private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---encoder---\n";
      PrintModelMetadata(os, meta_data);
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");

    // need to increase by 1 since the blank token is not included in computing
    // vocab_size in NeMo.
    vocab_size_ += 1;

    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
    SHERPA_ONNX_READ_META_DATA_STR(normalize_type_, "normalize_type");
    SHERPA_ONNX_READ_META_DATA(pred_rnn_layers_, "pred_rnn_layers");
    SHERPA_ONNX_READ_META_DATA(pred_hidden_, "pred_hidden");

    if (normalize_type_ == "NA") {
      normalize_type_ = "";
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                  &decoder_output_names_ptr_);
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    joiner_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(joiner_sess_.get(), &joiner_input_names_,
                  &joiner_input_names_ptr_);

    GetOutputNames(joiner_sess_.get(), &joiner_output_names_,
                  &joiner_output_names_ptr_);
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;
  std::unique_ptr<Ort::Session> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;

  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 8;
  std::string normalize_type_;
  int32_t pred_rnn_layers_ = -1;
  int32_t pred_hidden_ = -1;
};

OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OnlineTransducerNeMoModel::OnlineTransducerNeMoModel(
    AAssetManager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OnlineTransducerNeMoModel::~OnlineTransducerNeMoModel() = default;

int32_t ChunkLength() const { return window_size_; }

int32_t ChunkShift() const { return chunk_shift_; }

int32_t OnlineTransducerNeMoModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

int32_t OnlineTransducerNeMoModel::VocabSize() const {
  return impl_->VocabSize();
}

OrtAllocator *OnlineTransducerNeMoModel::Allocator() const {
  return impl_->Allocator();
}

std::string OnlineTransducerNeMoModel::FeatureNormalizationMethod() const {
  return impl_->FeatureNormalizationMethod();
}

}  // namespace sherpa_onnx