// sherpa-onnx/csrc/offline-tts-pocket-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-pocket-model.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static Ort::Value CreateZeroTensorLike(Ort::Session &sess, int32_t input_index,
                                       OrtAllocator *allocator) {
  auto type_info = sess.GetInputTypeInfo(input_index);
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType elem_type = tensor_info.GetElementType();
  std::vector<int64_t> shape = tensor_info.GetShape();

  // 3. Replace dynamic dims (-1) with 1
  for (auto &d : shape) {
    if (d < 0) {
      d = 1;
    }
  }

  Ort::Value v;
  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      v = Ort::Value::CreateTensor<float>(allocator, shape.data(),
                                          shape.size());
      Fill<float>(&v, 0);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      v = Ort::Value::CreateTensor<bool>(allocator, shape.data(), shape.size());
      Fill<bool>(&v, 0);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      v = Ort::Value::CreateTensor<int64_t>(allocator, shape.data(),
                                            shape.size());
      Fill<int64_t>(&v, 0);
      break;
    default:
      SHERPA_ONNX_LOGE("Unsupported tensor element type: %d", elem_type);
      SHERPA_ONNX_EXIT(-1);
  }

  return v;
}

class OfflineTtsPocketModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)) {
    lm_flow_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.pocket.lm_flow), sess_opts_);
    InitLmFlow(nullptr, 0);

    lm_main_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.pocket.lm_main), sess_opts_);
    InitLmMain(nullptr, 0);

    mimi_encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.pocket.encoder), sess_opts_);
    InitMimiEncoder(nullptr, 0);

    mimi_decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.pocket.decoder), sess_opts_);
    InitMimiDecoder(nullptr, 0);

    text_conditioner_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.pocket.text_conditioner),
        sess_opts_);
    InitTextConditioner(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)) {
    {
      auto buf = ReadFile(mgr, config.pocket.lm_flow);
      InitLmFlow(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.pocket.lm_main);
      InitLmMain(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.pocket.encoder);
      InitMimiEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.pocket.decoder);
      InitMimiDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.pocket.text_conditioner);
      InitTextConditioner(buf.data(), buf.size());
    }
  }

  PocketLmMainState GetLmMainInitState() {
    PocketLmMainState s;
    s.values.reserve(lm_main_init_states_.values.size());
    for (auto &v : lm_main_init_states_.values) {
      s.values.push_back(View(&v));
    }
    return s;
  }

  PocketMimiDecoderState GetMimiDecoderInitState() {
    PocketMimiDecoderState s;
    s.values.reserve(mimi_decoder_init_states_.values.size());
    for (auto &v : mimi_decoder_init_states_.values) {
      s.values.push_back(View(&v));
    }

    return s;
  }

  Ort::Value RunMimiEncoder(Ort::Value audio) const {
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(audio));

    auto outputs = mimi_encoder_sess_->Run(
        {}, mimi_encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        mimi_encoder_output_names_ptr_.data(),
        mimi_encoder_output_names_ptr_.size());

    return std::move(outputs[0]);
  }

  Ort::Value RunTextConditioner(Ort::Value text_tokens) const {
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(text_tokens));

    auto outputs = text_conditioner_sess_->Run(
        Ort::RunOptions{nullptr}, text_conditioner_input_names_ptr_.data(),
        inputs.data(), inputs.size(), text_conditioner_output_names_ptr_.data(),
        text_conditioner_output_names_ptr_.size());

    return std::move(outputs[0]);
  }

  std::tuple<Ort::Value, Ort::Value, PocketLmMainState> RunLmMain(
      Ort::Value seq, Ort::Value embeddings, PocketLmMainState state) const {
    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + state.values.size());

    inputs.push_back(std::move(seq));
    inputs.push_back(std::move(embeddings));

    for (auto &v : state.values) {
      inputs.push_back(std::move(v));
    }

    auto outputs = lm_main_sess_->Run(
        Ort::RunOptions{nullptr}, lm_main_input_names_ptr_.data(),
        inputs.data(), inputs.size(), lm_main_output_names_ptr_.data(),
        lm_main_output_names_ptr_.size());

    PocketLmMainState new_state;
    new_state.values.reserve(outputs.size() - 2);
    for (size_t i = 2; i < outputs.size(); ++i) {
      new_state.values.push_back(std::move(outputs[i]));
    }

    return {std::move(outputs[0]), std::move(outputs[1]), std::move(new_state)};
  }

  Ort::Value RunLmFlow(Ort::Value c, Ort::Value s, Ort::Value t,
                       Ort::Value x) const {
    std::vector<Ort::Value> inputs;
    inputs.reserve(4);
    inputs.push_back(std::move(c));
    inputs.push_back(std::move(s));
    inputs.push_back(std::move(t));
    inputs.push_back(std::move(x));

    auto outputs = lm_flow_sess_->Run(
        {}, lm_flow_input_names_ptr_.data(), inputs.data(), inputs.size(),
        lm_flow_output_names_ptr_.data(), lm_flow_output_names_ptr_.size());

    return std::move(outputs[0]);
  }

  std::pair<Ort::Value, PocketMimiDecoderState> RunMimiDecoder(
      Ort::Value latent, PocketMimiDecoderState state) const {
    std::vector<Ort::Value> inputs;
    inputs.reserve(1 + state.values.size());

    inputs.push_back(std::move(latent));
    for (auto &v : state.values) {
      inputs.push_back(std::move(v));
    }

    auto outputs = mimi_decoder_sess_->Run(
        {}, mimi_decoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        mimi_decoder_output_names_ptr_.data(),
        mimi_decoder_output_names_ptr_.size());

    PocketMimiDecoderState new_state;
    new_state.values.reserve(outputs.size() - 1);
    for (size_t i = 1; i < outputs.size(); ++i) {
      new_state.values.push_back(std::move(outputs[i]));
    }

    return {std::move(outputs[0]), std::move(new_state)};
  }

  OrtAllocator *Allocator() { return allocator_; }

 private:
  void InitLmFlow(void *model_data, size_t model_data_length) {
    if (model_data) {
      lm_flow_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!lm_flow_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize lm flow session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(lm_flow_sess_.get(), &lm_flow_input_names_,
                  &lm_flow_input_names_ptr_);

    GetOutputNames(lm_flow_sess_.get(), &lm_flow_output_names_,
                   &lm_flow_output_names_ptr_);
  }

  void InitLmMain(void *model_data, size_t model_data_length) {
    if (model_data) {
      lm_main_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!lm_main_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize lm main session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(lm_main_sess_.get(), &lm_main_input_names_,
                  &lm_main_input_names_ptr_);

    GetOutputNames(lm_main_sess_.get(), &lm_main_output_names_,
                   &lm_main_output_names_ptr_);

    lm_main_init_states_.values.reserve(lm_main_input_names_.size() - 2);
    for (size_t i = 2; i < lm_main_input_names_.size(); ++i) {
      lm_main_init_states_.values.push_back(
          CreateZeroTensorLike(*lm_main_sess_, i, allocator_));
    }
  }

  void InitMimiEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      mimi_encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!mimi_encoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize mimi encoder session outside "
          "of this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(mimi_encoder_sess_.get(), &mimi_encoder_input_names_,
                  &mimi_encoder_input_names_ptr_);

    GetOutputNames(mimi_encoder_sess_.get(), &mimi_encoder_output_names_,
                   &mimi_encoder_output_names_ptr_);
  }

  void InitMimiDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      mimi_decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!mimi_decoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize mimi decoder session outside "
          "of this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(mimi_decoder_sess_.get(), &mimi_decoder_input_names_,
                  &mimi_decoder_input_names_ptr_);

    GetOutputNames(mimi_decoder_sess_.get(), &mimi_decoder_output_names_,
                   &mimi_decoder_output_names_ptr_);

    // init mimi_decoder_init_states_
    mimi_decoder_init_states_.values.reserve(mimi_decoder_input_names_.size() -
                                             1);
    for (size_t i = 1; i < mimi_decoder_input_names_.size(); ++i) {
      mimi_decoder_init_states_.values.push_back(
          CreateZeroTensorLike(*mimi_decoder_sess_, i, allocator_));
    }
  }

  void InitTextConditioner(void *model_data, size_t model_data_length) {
    if (model_data) {
      text_conditioner_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!text_conditioner_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize text conditioner session "
          "outside of this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(text_conditioner_sess_.get(), &text_conditioner_input_names_,
                  &text_conditioner_input_names_ptr_);

    GetOutputNames(text_conditioner_sess_.get(),
                   &text_conditioner_output_names_,
                   &text_conditioner_output_names_ptr_);
  }

 private:
  OfflineTtsModelConfig config_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> lm_main_sess_;
  std::unique_ptr<Ort::Session> lm_flow_sess_;
  std::unique_ptr<Ort::Session> mimi_decoder_sess_;
  std::unique_ptr<Ort::Session> mimi_encoder_sess_;
  std::unique_ptr<Ort::Session> text_conditioner_sess_;

  std::vector<std::string> lm_flow_input_names_;
  std::vector<const char *> lm_flow_input_names_ptr_;

  std::vector<std::string> lm_flow_output_names_;
  std::vector<const char *> lm_flow_output_names_ptr_;

  std::vector<std::string> lm_main_input_names_;
  std::vector<const char *> lm_main_input_names_ptr_;

  std::vector<std::string> lm_main_output_names_;
  std::vector<const char *> lm_main_output_names_ptr_;

  std::vector<std::string> mimi_encoder_input_names_;
  std::vector<const char *> mimi_encoder_input_names_ptr_;

  std::vector<std::string> mimi_encoder_output_names_;
  std::vector<const char *> mimi_encoder_output_names_ptr_;

  std::vector<std::string> mimi_decoder_input_names_;
  std::vector<const char *> mimi_decoder_input_names_ptr_;

  std::vector<std::string> mimi_decoder_output_names_;
  std::vector<const char *> mimi_decoder_output_names_ptr_;

  std::vector<std::string> text_conditioner_input_names_;
  std::vector<const char *> text_conditioner_input_names_ptr_;

  std::vector<std::string> text_conditioner_output_names_;
  std::vector<const char *> text_conditioner_output_names_ptr_;

  PocketLmMainState lm_main_init_states_;
  PocketMimiDecoderState mimi_decoder_init_states_;
};

OfflineTtsPocketModel::OfflineTtsPocketModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsPocketModel::OfflineTtsPocketModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsPocketModel::~OfflineTtsPocketModel() = default;

PocketLmMainState OfflineTtsPocketModel::GetLmMainInitState() const {
  return impl_->GetLmMainInitState();
}

PocketMimiDecoderState OfflineTtsPocketModel::GetMimiDecoderInitState() const {
  return impl_->GetMimiDecoderInitState();
}

Ort::Value OfflineTtsPocketModel::RunMimiEncoder(Ort::Value audio) const {
  return impl_->RunMimiEncoder(std::move(audio));
}

Ort::Value OfflineTtsPocketModel::RunTextConditioner(
    Ort::Value text_tokens) const {
  return impl_->RunTextConditioner(std::move(text_tokens));
}

std::tuple<Ort::Value, Ort::Value, PocketLmMainState>
OfflineTtsPocketModel::RunLmMain(Ort::Value seq, Ort::Value embeddings,
                                 PocketLmMainState state) const {
  return impl_->RunLmMain(std::move(seq), std::move(embeddings),
                          std::move(state));
}

Ort::Value OfflineTtsPocketModel::RunLmFlow(Ort::Value c, Ort::Value s,
                                            Ort::Value t, Ort::Value x) const {
  return impl_->RunLmFlow(std::move(c), std::move(s), std::move(t),
                          std::move(x));
}

std::pair<Ort::Value, PocketMimiDecoderState>
OfflineTtsPocketModel::RunMimiDecoder(Ort::Value latent,
                                      PocketMimiDecoderState state) const {
  return impl_->RunMimiDecoder(std::move(latent), std::move(state));
}

OrtAllocator *OfflineTtsPocketModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineTtsPocketModel::OfflineTtsPocketModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsPocketModel::OfflineTtsPocketModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
