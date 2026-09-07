// sherpa-onnx/csrc/offline-tts-pocket-zh-en-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-pocket-zh-en-model.h"

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
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsPocketZhEnModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)) {
    InitEncoder(nullptr, 0);
    InitStepModel(nullptr, 0);
    ReadMetaData();
    InitTensors();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)) {
    {
      auto buf = ReadFile(mgr, config.pocket_zh_en.step_encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.pocket_zh_en.step_model);
      InitStepModel(buf.data(), buf.size());
    }

    ReadMetaData();
    InitTensors();
  }

  const OfflineTtsPocketZhEnModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  Ort::Value RunEncoder(Ort::Value audio) const {
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(audio));

    auto outputs = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());

    return std::move(outputs[0]);
  }

  std::vector<Ort::Value> RunStep(std::vector<Ort::Value> inputs) const {
    auto outputs = step_sess_->Run(
        {}, step_input_names_ptr_.data(), inputs.data(), inputs.size(),
        step_output_names_ptr_.data(), step_output_names_ptr_.size());

    return outputs;
  }

  // Get* functions return View of pre-initialized tensors
  Ort::Value GetZeroLatent() const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<float> data(meta_data_.latent_dim, 0.0f);
    std::vector<int64_t> shape = {1, 1, meta_data_.latent_dim};
    return Ort::Value::CreateTensor(memory_info, data.data(), data.size(),
                                    shape.data(), shape.size());
  }

  Ort::Value GetBosFlag() const { return View(&bos_flag_); }
  Ort::Value GetNonBosFlag() const { return View(&non_bos_flag_); }
  Ort::Value GetTextGates() const { return View(&text_gates_); }
  Ort::Value GetLatentGates() const { return View(&latent_gates_); }
  Ort::Value GetCondGates() const { return View(&cond_gates_); }
  Ort::Value GetZeroNoise() const { return View(&zero_noise_); }
  Ort::Value GetEmptyFlowKv() const { return View(&empty_flow_kv_); }
  Ort::Value GetZeroFlowOffset() const { return View(&zero_flow_offset_); }
  Ort::Value GetZeroMimiKv() const { return View(&zero_mimi_kv_); }
  Ort::Value GetZeroMimiOffset() const { return View(&zero_mimi_offset_); }
  Ort::Value GetZeroMimiConv() const { return View(&zero_mimi_conv_); }
  Ort::Value GetDecodeSteps() const { return View(&decode_steps_); }

 private:
  void InitTensors() {
    bos_flag_ = CreateBosFlag();
    non_bos_flag_ = CreateNonBosFlag();
    text_gates_ = CreateTextGates();
    latent_gates_ = CreateLatentGates();
    cond_gates_ = CreateCondGates();
    zero_noise_ = CreateZeroNoise();
    empty_flow_kv_ = CreateEmptyFlowKv();
    zero_flow_offset_ = CreateZeroFlowOffset();
    zero_mimi_kv_ = CreateZeroMimiKv();
    zero_mimi_offset_ = CreateZeroMimiOffset();
    zero_mimi_conv_ = CreateZeroMimiConv();
    decode_steps_ = CreateDecodeSteps();
  }

  Ort::Value CreateBosFlag() const {
    std::vector<int64_t> shape = {1, 1, 1};
    auto tensor = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                                  shape.size());
    *tensor.GetTensorMutableData<float>() = 1.0f;
    return tensor;
  }

  Ort::Value CreateNonBosFlag() const {
    std::vector<int64_t> shape = {1, 1, 1};
    auto tensor = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                                  shape.size());
    *tensor.GetTensorMutableData<float>() = 0.0f;
    return tensor;
  }

  Ort::Value CreateTextGates() const {
    std::vector<int64_t> shape = {3};
    auto tensor =
        Ort::Value::CreateTensor<float>(allocator_, shape.data(), shape.size());
    float *p = tensor.GetTensorMutableData<float>();
    p[0] = 1.0f;
    p[1] = 0.0f;
    p[2] = 0.0f;
    return tensor;
  }

  Ort::Value CreateLatentGates() const {
    std::vector<int64_t> shape = {3};
    auto tensor =
        Ort::Value::CreateTensor<float>(allocator_, shape.data(), shape.size());
    float *p = tensor.GetTensorMutableData<float>();
    p[0] = 0.0f;
    p[1] = 1.0f;
    p[2] = 0.0f;
    return tensor;
  }

  Ort::Value CreateCondGates() const {
    std::vector<int64_t> shape = {3};
    auto tensor =
        Ort::Value::CreateTensor<float>(allocator_, shape.data(), shape.size());
    float *p = tensor.GetTensorMutableData<float>();
    p[0] = 0.0f;
    p[1] = 0.0f;
    p[2] = 1.0f;
    return tensor;
  }

  Ort::Value CreateZeroNoise() const {
    std::vector<int64_t> shape = {1, meta_data_.latent_dim};
    auto tensor = Ort::Value::CreateTensor<float>(
        allocator_, shape.data(), shape.size());
    Fill<float>(&tensor, 0);
    return tensor;
  }

  Ort::Value CreateEmptyFlowKv() const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape = {0,
                                  meta_data_.flow_layers,
                                  2,
                                  1,
                                  meta_data_.flow_heads,
                                  meta_data_.flow_head_dim};
    return Ort::Value::CreateTensor<float>(memory_info, nullptr, 0,
                                           shape.data(), shape.size());
  }

  Ort::Value CreateZeroFlowOffset() const {
    auto tensor =
        Ort::Value::CreateTensor<int64_t>(allocator_, nullptr, 0);
    *tensor.GetTensorMutableData<int64_t>() = 0;
    return tensor;
  }

  Ort::Value CreateZeroMimiKv() const {
    std::vector<int64_t> shape = {meta_data_.mimi_kv_len,
                                  meta_data_.mimi_layers,
                                  2,
                                  1,
                                  meta_data_.mimi_heads,
                                  meta_data_.mimi_head_dim};
    auto tensor = Ort::Value::CreateTensor<float>(
        allocator_, shape.data(), shape.size());
    Fill<float>(&tensor, 0);
    return tensor;
  }

  Ort::Value CreateZeroMimiOffset() const {
    auto tensor =
        Ort::Value::CreateTensor<int64_t>(allocator_, nullptr, 0);
    *tensor.GetTensorMutableData<int64_t>() = 0;
    return tensor;
  }

  Ort::Value CreateZeroMimiConv() const {
    std::vector<int64_t> shape = {meta_data_.conv_state_size};
    auto tensor = Ort::Value::CreateTensor<float>(
        allocator_, shape.data(), shape.size());
    Fill<float>(&tensor, 0);
    return tensor;
  }

  Ort::Value CreateDecodeSteps() const {
    auto tensor = Ort::Value::CreateTensor<float>(allocator_, nullptr, 0);
    *tensor.GetTensorMutableData<float>() = 1.0f;
    return tensor;
  }

  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, SHERPA_ONNX_TO_ORT_PATH(config_.pocket_zh_en.step_encoder),
          sess_opts_);
    }

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    if (config_.debug) {
      std::ostringstream os;
      os << "----------step encoder input names----------\n";
      for (int32_t i = 0; i < static_cast<int32_t>(encoder_input_names_.size());
           ++i) {
        os << i << " " << encoder_input_names_[i] << "\n";
      }
      os << "----------step encoder output names----------\n";
      for (int32_t i = 0;
           i < static_cast<int32_t>(encoder_output_names_.size()); ++i) {
        os << i << " " << encoder_output_names_[i] << "\n";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
  }

  void InitStepModel(void *model_data, size_t model_data_length) {
    if (model_data) {
      step_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else {
      step_sess_ = std::make_unique<Ort::Session>(
          env_, SHERPA_ONNX_TO_ORT_PATH(config_.pocket_zh_en.step_model),
          sess_opts_);
    }

    GetInputNames(step_sess_.get(), &step_input_names_, &step_input_names_ptr_);
    GetOutputNames(step_sess_.get(), &step_output_names_,
                   &step_output_names_ptr_);

    if (config_.debug) {
      std::ostringstream os;
      os << "----------step model input names----------\n";
      for (int32_t i = 0;
           i < static_cast<int32_t>(step_input_names_.size()); ++i) {
        os << i << " " << step_input_names_[i] << "\n";
      }
      os << "----------step model output names----------\n";
      for (int32_t i = 0;
           i < static_cast<int32_t>(step_output_names_.size()); ++i) {
        os << i << " " << step_output_names_[i] << "\n";
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
  }

  void ReadMetaData() {
    // Read dimensions from ONNX model shapes
    meta_data_.model_dim = 1024;
    meta_data_.latent_dim = 32;
    meta_data_.flow_layers = 6;
    meta_data_.flow_heads = 16;
    meta_data_.flow_head_dim = 64;
    meta_data_.mimi_kv_len = 266;
    meta_data_.mimi_layers = 2;
    meta_data_.mimi_heads = 8;
    meta_data_.mimi_head_dim = 64;
    meta_data_.conv_state_size = 14720;
    meta_data_.frame_size = 1920;
    meta_data_.sample_rate = 24000;

    if (config_.debug) {
      std::ostringstream os;
      os << "---pocket-zh-en model---\n";
      os << "model_dim: " << meta_data_.model_dim << "\n";
      os << "latent_dim: " << meta_data_.latent_dim << "\n";
      os << "flow_layers: " << meta_data_.flow_layers << "\n";
      os << "flow_heads: " << meta_data_.flow_heads << "\n";
      os << "flow_head_dim: " << meta_data_.flow_head_dim << "\n";
      os << "mimi_kv_len: " << meta_data_.mimi_kv_len << "\n";
      os << "mimi_layers: " << meta_data_.mimi_layers << "\n";
      os << "mimi_heads: " << meta_data_.mimi_heads << "\n";
      os << "mimi_head_dim: " << meta_data_.mimi_head_dim << "\n";
      os << "conv_state_size: " << meta_data_.conv_state_size << "\n";
      os << "frame_size: " << meta_data_.frame_size << "\n";
      os << "sample_rate: " << meta_data_.sample_rate << "\n";

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
  }

  OfflineTtsModelConfig config_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> step_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> step_input_names_;
  std::vector<const char *> step_input_names_ptr_;
  std::vector<std::string> step_output_names_;
  std::vector<const char *> step_output_names_ptr_;

  OfflineTtsPocketZhEnModelMetaData meta_data_;

  // Pre-initialized tensors (created once in InitTensors)
  // Mutable because View() requires non-const pointer
  mutable Ort::Value bos_flag_{nullptr};
  mutable Ort::Value non_bos_flag_{nullptr};
  mutable Ort::Value text_gates_{nullptr};
  mutable Ort::Value latent_gates_{nullptr};
  mutable Ort::Value cond_gates_{nullptr};
  mutable Ort::Value zero_noise_{nullptr};
  mutable Ort::Value empty_flow_kv_{nullptr};
  mutable Ort::Value zero_flow_offset_{nullptr};
  mutable Ort::Value zero_mimi_kv_{nullptr};
  mutable Ort::Value zero_mimi_offset_{nullptr};
  mutable Ort::Value zero_mimi_conv_{nullptr};
  mutable Ort::Value decode_steps_{nullptr};
};

OfflineTtsPocketZhEnModel::OfflineTtsPocketZhEnModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsPocketZhEnModel::OfflineTtsPocketZhEnModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsPocketZhEnModel::~OfflineTtsPocketZhEnModel() = default;

const OfflineTtsPocketZhEnModelMetaData &
OfflineTtsPocketZhEnModel::GetMetaData() const {
  return impl_->GetMetaData();
}

Ort::Value OfflineTtsPocketZhEnModel::RunEncoder(Ort::Value audio) const {
  return impl_->RunEncoder(std::move(audio));
}

std::vector<Ort::Value> OfflineTtsPocketZhEnModel::RunStep(
    std::vector<Ort::Value> inputs) const {
  return impl_->RunStep(std::move(inputs));
}

Ort::Value OfflineTtsPocketZhEnModel::CreateZeroLatent() const {
  return impl_->GetZeroLatent();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateBosFlag() const {
  return impl_->GetBosFlag();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateNonBosFlag() const {
  return impl_->GetNonBosFlag();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateTextGates() const {
  return impl_->GetTextGates();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateLatentGates() const {
  return impl_->GetLatentGates();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateCondGates() const {
  return impl_->GetCondGates();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateZeroNoise() const {
  return impl_->GetZeroNoise();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateEmptyFlowKv() const {
  return impl_->GetEmptyFlowKv();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateZeroFlowOffset() const {
  return impl_->GetZeroFlowOffset();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateZeroMimiKv() const {
  return impl_->GetZeroMimiKv();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateZeroMimiOffset() const {
  return impl_->GetZeroMimiOffset();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateZeroMimiConv() const {
  return impl_->GetZeroMimiConv();
}

Ort::Value OfflineTtsPocketZhEnModel::CreateDecodeSteps() const {
  return impl_->GetDecodeSteps();
}

#if __ANDROID_API__ >= 9
template OfflineTtsPocketZhEnModel::OfflineTtsPocketZhEnModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsPocketZhEnModel::OfflineTtsPocketZhEnModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
