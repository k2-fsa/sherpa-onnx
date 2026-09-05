// sherpa-onnx/csrc/offline-qwen3-asr-model.cc
//
// Copyright (c)  2026  zengyw

#include "sherpa-onnx/csrc/offline-qwen3-asr-model.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

constexpr int32_t kDecoderCachePositionInputIndex = 3;
constexpr int32_t kDecoderPastKvInputStartIndex = 4;

inline size_t NumelFromShape(const std::vector<int64_t> &shape) {
  if (shape.empty()) return 0;
  size_t n = 1;
  for (auto d : shape) {
    if (d <= 0) return 0;
    n *= static_cast<size_t>(d);
  }
  return n;
}

inline size_t ElemBytesFromTensorType(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return 2;
    default:
      SHERPA_ONNX_LOGE("Unsupported tensor element type: %d",
                       static_cast<int>(t));
      SHERPA_ONNX_EXIT(-1);
      return 0;
  }
}

template <typename T>
Ort::Value AllocTensor(OrtAllocator *alloc, const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor<T>(alloc, shape.data(), shape.size());
}

inline Ort::Value AllocTensorFp16(OrtAllocator *alloc,
                                  const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor(alloc, shape.data(), shape.size(),
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
}

inline Ort::Value AllocTensorUInt16(OrtAllocator *alloc,
                                    const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor(alloc, shape.data(), shape.size(),
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
}

inline Ort::Value AllocTensorByElemType(OrtAllocator *alloc,
                                        const std::vector<int64_t> &shape,
                                        ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return AllocTensor<float>(alloc, shape);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return AllocTensorFp16(alloc, shape);

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return AllocTensorUInt16(alloc, shape);

    default:
      SHERPA_ONNX_LOGE("AllocTensorByElemType: unsupported elem_type=%d",
                       static_cast<int>(t));
      SHERPA_ONNX_EXIT(-1);
      return AllocTensor<float>(alloc, shape);
  }
}

inline ONNXTensorElementDataType GetSessionInputElemType(Ort::Session *sess,
                                                         size_t input_index) {
  auto ti = sess->GetInputTypeInfo(input_index);
  auto t = ti.GetTensorTypeAndShapeInfo();
  return static_cast<ONNXTensorElementDataType>(t.GetElementType());
}

inline bool IsCudaProvider(const std::string &provider) {
  std::string p = provider;
  std::transform(p.begin(), p.end(), p.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return p == "cuda" || (p.size() > 4 && p.find("cuda") == 0);
}

}  // namespace

class OfflineQwen3ASRModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "qwen3-asr"),
        sess_opts_conv_(GetSessionOptions(config)),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_decoder_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.qwen3_asr;

    conv_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.conv_frontend), sess_opts_conv_);

    InitConvFrontend(nullptr, 0);

    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.encoder), sess_opts_encoder_);

    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(c.decoder), sess_opts_decoder_);

    InitDecoder(nullptr, 0);

    InitIoBindingConfig();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "qwen3-asr"),
        sess_opts_conv_(GetSessionOptions(config)),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_decoder_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.qwen3_asr;

    {
      auto buf = ReadFile(mgr, c.conv_frontend);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read qwen3_asr.conv_frontend: %s",
                         c.conv_frontend.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      InitConvFrontend(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(mgr, c.encoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read qwen3_asr.encoder: %s",
                         c.encoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      InitEncoder(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(mgr, c.decoder);
      if (buf.empty()) {
        SHERPA_ONNX_LOGE("Failed to read qwen3_asr.decoder: %s",
                         c.decoder.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      InitDecoder(buf.data(), buf.size());
    }

    InitIoBindingConfig();
  }

 private:
  void InitIoBindingConfig() {
    use_cuda_iobinding_ =
        (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
  }

  void InitSessionIo(Ort::Session *sess, std::vector<std::string> *input_names,
                     std::vector<const char *> *input_names_ptr,
                     std::vector<std::string> *output_names,
                     std::vector<const char *> *output_names_ptr) {
    GetInputNames(sess, input_names, input_names_ptr);
    GetOutputNames(sess, output_names, output_names_ptr);
  }

  void InitConvFrontend(void *model_data, size_t model_data_length) {
    if (model_data) {
      conv_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_conv_);
    } else if (!conv_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize conv session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    InitSessionIo(conv_sess_.get(), &conv_input_names_, &conv_input_names_ptr_,
                  &conv_output_names_, &conv_output_names_ptr_);

    if (conv_input_names_.size() != 1) {
      SHERPA_ONNX_LOGE("conv_frontend must have exactly 1 input, got %zu",
                       conv_input_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }

    if (conv_output_names_.size() != 1) {
      SHERPA_ONNX_LOGE("conv_frontend must have exactly 1 output, got %zu",
                       conv_output_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_encoder_);
    } else if (!encoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize encoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    InitEncoderSession();
  }

  void InitEncoderSession() {
    InitSessionIo(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_, &encoder_output_names_,
                  &encoder_output_names_ptr_);

    if (encoder_input_names_.size() != 2) {
      SHERPA_ONNX_LOGE("encoder must have exactly 2 inputs, got %zu",
                       encoder_input_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }

    if (encoder_output_names_.size() != 1) {
      SHERPA_ONNX_LOGE("encoder must have exactly 1 output, got %zu",
                       encoder_output_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_decoder_);
    } else if (!decoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize decoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    InitDecoderSession();
  }

  void InitDecoderSession() {
    InitSessionIo(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_, &decoder_output_names_,
                  &decoder_output_names_ptr_);

    size_t num_dec_outputs = decoder_output_names_.size();
    if (num_dec_outputs < 1 || ((num_dec_outputs - 1) % 2) != 0) {
      SHERPA_ONNX_LOGE(
          "Decoder model outputs mismatch: expected 1+2*num_layers, got %zu",
          num_dec_outputs);
      SHERPA_ONNX_EXIT(-1);
    }
    num_layers_ = static_cast<int32_t>((num_dec_outputs - 1) / 2);

    int32_t expected_inputs = 4 + 2 * num_layers_;
    int32_t actual_inputs = static_cast<int32_t>(decoder_input_names_.size());
    if (actual_inputs != expected_inputs) {
      SHERPA_ONNX_LOGE(
          "Decoder model inputs mismatch: expected %d (=4+2*num_layers) got %d",
          expected_inputs, actual_inputs);
      SHERPA_ONNX_EXIT(-1);
    }

    auto past_key_ti =
        decoder_sess_->GetInputTypeInfo(kDecoderPastKvInputStartIndex);
    past_key_shape_tpl_ = past_key_ti.GetTensorTypeAndShapeInfo().GetShape();
    if (past_key_shape_tpl_.size() >= 2 && past_key_shape_tpl_[1] > 0) {
      max_total_len_ = static_cast<int32_t>(past_key_shape_tpl_[1]);
    } else {
      max_total_len_ = config_.qwen3_asr.max_total_len;
    }
    if (max_total_len_ <= 0) {
      SHERPA_ONNX_LOGE(
          "Failed to determine max_total_len (past_key dim1 is dynamic or "
          "invalid; set --qwen3-asr-max-total-len)");
      SHERPA_ONNX_EXIT(-1);
    }

    kv_in_type_ = GetSessionInputElemType(decoder_sess_.get(),
                                          kDecoderPastKvInputStartIndex);
    kv_in_type_v_ = GetSessionInputElemType(decoder_sess_.get(),
                                            kDecoderPastKvInputStartIndex + 1);
    if (!(kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
      SHERPA_ONNX_LOGE("Decoder past_key elem_type=%d not supported",
                       (int)kv_in_type_);
      SHERPA_ONNX_EXIT(-1);
    }
    if (!(kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
      SHERPA_ONNX_LOGE("Decoder past_value elem_type=%d not supported",
                       (int)kv_in_type_v_);
      SHERPA_ONNX_EXIT(-1);
    }
  }

 public:
  Ort::Value ForwardConvFrontend(Ort::Value input_features) {
    if (use_cuda_iobinding_) {
      Ort::IoBinding binding(*conv_sess_);
      binding.BindInput(conv_input_names_ptr_[0], input_features);
      binding.BindOutput(conv_output_names_ptr_[0], *cuda_mem_info_);
      binding.SynchronizeInputs();
      conv_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      auto outs = binding.GetOutputValues();
      if (outs.empty()) {
        SHERPA_ONNX_LOGE("ForwardConvFrontend: empty outputs");
        SHERPA_ONNX_EXIT(-1);
      }
      return std::move(outs[0]);
    }

    std::array<Ort::Value, 1> inputs = {std::move(input_features)};
    auto outputs = conv_sess_->Run(
        {}, conv_input_names_ptr_.data(), inputs.data(), 1,
        conv_output_names_ptr_.data(), conv_output_names_ptr_.size());
    if (outputs.empty()) {
      SHERPA_ONNX_LOGE("ForwardConvFrontend: empty outputs");
      SHERPA_ONNX_EXIT(-1);
    }
    return std::move(outputs[0]);
  }

  Ort::Value ForwardEncoder(Ort::Value conv_output,
                            Ort::Value feature_attention_mask) {
    auto mask_info = feature_attention_mask.GetTensorTypeAndShapeInfo();
    auto mask_type =
        static_cast<ONNXTensorElementDataType>(mask_info.GetElementType());
    if (mask_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
      SHERPA_ONNX_LOGE(
          "ForwardEncoder: feature_attention_mask must be bool, got "
          "elem_type=%d",
          static_cast<int>(mask_type));
      SHERPA_ONNX_EXIT(-1);
    }

    if (use_cuda_iobinding_) {
      Ort::IoBinding binding(*encoder_sess_);
      binding.BindInput(encoder_input_names_ptr_[0], conv_output);
      binding.BindInput(encoder_input_names_ptr_[1], feature_attention_mask);
      binding.BindOutput(encoder_output_names_ptr_[0], cpu_mem_info_);
      binding.SynchronizeInputs();
      encoder_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      auto outs = binding.GetOutputValues();
      if (outs.empty()) {
        SHERPA_ONNX_LOGE("ForwardEncoder: empty outputs");
        SHERPA_ONNX_EXIT(-1);
      }
      return std::move(outs[0]);
    }

    std::array<Ort::Value, 2> inputs = {std::move(conv_output),
                                        std::move(feature_attention_mask)};
    auto outputs = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    if (outputs.empty()) {
      SHERPA_ONNX_LOGE("ForwardEncoder: empty outputs");
      SHERPA_ONNX_EXIT(-1);
    }
    return std::move(outputs[0]);
  }

  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLM(Ort::Value input_ids, Ort::Value audio_features,
             Ort::Value attention_mask, const Ort::Value &cache_position,
             const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv) {
    if (static_cast<int32_t>(cache_kv.size()) != num_layers_) {
      SHERPA_ONNX_LOGE("ForwardLLM: cache_kv size (%zu) != num_layers (%d)",
                       cache_kv.size(), num_layers_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (!audio_features.IsTensor()) {
      SHERPA_ONNX_LOGE("ForwardLLM: audio_features is not a tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    auto af_info = audio_features.GetTensorTypeAndShapeInfo();
    auto af_shape = af_info.GetShape();
    if (af_shape.size() != 3) {
      SHERPA_ONNX_LOGE(
          "ForwardLLM: audio_features must be 3D tensor [B, A, H]");
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(4 + 2 * cache_kv.size());
    inputs.push_back(std::move(input_ids));
    inputs.push_back(std::move(audio_features));
    inputs.push_back(std::move(attention_mask));
    inputs.push_back(View(const_cast<Ort::Value *>(&cache_position)));

    for (const auto &kv : cache_kv) {
      inputs.push_back(View(const_cast<Ort::Value *>(&kv.first)));
      inputs.push_back(View(const_cast<Ort::Value *>(&kv.second)));
    }

    std::vector<const char *> input_names_ptr;
    input_names_ptr.reserve(4 + 2 * cache_kv.size());
    input_names_ptr.push_back(decoder_input_names_ptr_[0]);
    input_names_ptr.push_back(decoder_input_names_ptr_[1]);
    input_names_ptr.push_back(decoder_input_names_ptr_[2]);
    input_names_ptr.push_back(
        decoder_input_names_ptr_[kDecoderCachePositionInputIndex]);
    for (size_t i = 0; i < cache_kv.size(); ++i) {
      input_names_ptr.push_back(
          decoder_input_names_ptr_[kDecoderPastKvInputStartIndex + 2 * i]);
      input_names_ptr.push_back(
          decoder_input_names_ptr_[kDecoderPastKvInputStartIndex + 2 * i + 1]);
    }

    std::vector<Ort::Value> outputs;

    if (use_cuda_iobinding_) {
      Ort::IoBinding binding(*decoder_sess_);
      for (size_t i = 0; i < inputs.size(); ++i) {
        binding.BindInput(input_names_ptr[i], inputs[i]);
      }
      binding.BindOutput(decoder_output_names_ptr_[0], cpu_mem_info_);
      for (size_t i = 1; i < decoder_output_names_ptr_.size(); ++i) {
        binding.BindOutput(decoder_output_names_ptr_[i], cpu_mem_info_);
      }
      binding.SynchronizeInputs();
      decoder_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      outputs = binding.GetOutputValues();
    } else {
      outputs = decoder_sess_->Run(
          {}, input_names_ptr.data(), inputs.data(), inputs.size(),
          decoder_output_names_ptr_.data(), decoder_output_names_ptr_.size());
    }

    if (outputs.empty()) {
      SHERPA_ONNX_LOGE("ForwardLLM: empty outputs");
      SHERPA_ONNX_EXIT(-1);
    }

    Ort::Value logits = std::move(outputs[0]);

    if (!logits.IsTensor()) {
      SHERPA_ONNX_LOGE("ForwardLLM: logits is not a tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    auto logits_info = logits.GetTensorTypeAndShapeInfo();
    auto logits_type =
        static_cast<ONNXTensorElementDataType>(logits_info.GetElementType());
    if (outputs.size() < 1 || ((outputs.size() - 1) % 2) != 0) {
      SHERPA_ONNX_LOGE(
          "ForwardLLM: output count mismatch: expect 1+2*num_layers, got %zu",
          outputs.size());
      SHERPA_ONNX_EXIT(-1);
    }
    if (logits_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        logits_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
        logits_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
      SHERPA_ONNX_LOGE(
          "ForwardLLM: logits must be float32 or float16, got elem_type=%d",
          (int)logits_type);
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t inferred_layers = static_cast<int32_t>((outputs.size() - 1) / 2);
    if (inferred_layers != num_layers_) {
      SHERPA_ONNX_LOGE(
          "ForwardLLM: KV outputs layers mismatch: expected=%d, got=%d",
          num_layers_, inferred_layers);
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<std::pair<Ort::Value, Ort::Value>> kv_outputs;
    kv_outputs.reserve(num_layers_);
    for (int32_t i = 0; i < num_layers_; ++i) {
      kv_outputs.emplace_back(std::move(outputs[1 + 2 * i]),
                              std::move(outputs[1 + 2 * i + 1]));
    }

    return {std::move(logits), std::move(kv_outputs)};
  }

  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(
      int64_t batch) {
    std::vector<std::pair<Ort::Value, Ort::Value>> kv_cache;
    kv_cache.reserve(num_layers_);

    auto &tpl = past_key_shape_tpl_;
    if (tpl.size() < 4) {
      SHERPA_ONNX_LOGE("Invalid KV cache shape template, expected >=4 dims");
      SHERPA_ONNX_EXIT(-1);
    }
    int64_t kv_h = tpl[2];
    int64_t hd = tpl[3];
    std::vector<int64_t> key_shape = {
        batch, static_cast<int64_t>(max_total_len_), kv_h, hd};
    std::vector<int64_t> value_shape = key_shape;

    size_t key_numel = NumelFromShape(key_shape);
    size_t value_numel = NumelFromShape(value_shape);

    for (int32_t i = 0; i < num_layers_; ++i) {
      Ort::Value key_tensor =
          AllocTensorByElemType(allocator_, key_shape, kv_in_type_);
      Ort::Value value_tensor =
          AllocTensorByElemType(allocator_, value_shape, kv_in_type_v_);

      if (key_numel > 0) {
        if (kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          std::memset(key_tensor.GetTensorMutableData<float>(), 0,
                      key_numel * sizeof(float));
        } else {
          std::memset(key_tensor.GetTensorMutableData<uint16_t>(), 0,
                      key_numel * sizeof(uint16_t));
        }
      }

      if (value_numel > 0) {
        if (kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          std::memset(value_tensor.GetTensorMutableData<float>(), 0,
                      value_numel * sizeof(float));
        } else {
          std::memset(value_tensor.GetTensorMutableData<uint16_t>(), 0,
                      value_numel * sizeof(uint16_t));
        }
      }

      kv_cache.emplace_back(std::move(key_tensor), std::move(value_tensor));
    }
    return kv_cache;
  }

  void ApplyKvDeltaInplace(
      std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
      const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
      const Ort::Value &cache_position) const {
    if (!cache_kv || cache_kv->size() != static_cast<size_t>(num_layers_) ||
        kv_delta.size() != static_cast<size_t>(num_layers_)) {
      SHERPA_ONNX_LOGE(
          "ApplyKvDeltaInplace: invalid kv sizes: cache=%zu delta=%zu",
          cache_kv ? cache_kv->size() : 0, kv_delta.size());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!cache_position.IsTensor()) {
      SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: cache_position is not a tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    auto pos_info = cache_position.GetTensorTypeAndShapeInfo();
    if (pos_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      SHERPA_ONNX_LOGE(
          "ApplyKvDeltaInplace: cache_position must be int64, got %d",
          static_cast<int32_t>(pos_info.GetElementType()));
      SHERPA_ONNX_EXIT(-1);
    }
    auto pos_shape = pos_info.GetShape();
    if (pos_shape.size() != 1) {
      SHERPA_ONNX_LOGE(
          "ApplyKvDeltaInplace: cache_position must be 1-D, got %zu-D",
          pos_shape.size());
      SHERPA_ONNX_EXIT(-1);
    }
    int64_t S = pos_shape.empty() ? 0 : pos_shape[0];
    if (S <= 0) {
      SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: cache_position has invalid shape");
      SHERPA_ONNX_EXIT(-1);
    }

    const int64_t *pos_data = cache_position.GetTensorData<int64_t>();
    int64_t pos0 = pos_data[0];

    if (pos0 < 0) {
      SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: pos0 < 0 (%d)",
                       static_cast<int32_t>(pos0));
      SHERPA_ONNX_EXIT(-1);
    }
    if (pos0 + S > max_total_len_) {
      SHERPA_ONNX_LOGE(
          "ApplyKvDeltaInplace: pos0+S exceeds max_total_len_ (%d + %d > "
          "%d), clamping S",
          static_cast<int32_t>(pos0), static_cast<int32_t>(S), max_total_len_);
      S = max_total_len_ - pos0;
      if (S <= 0) return;
    }

    for (int32_t layer = 0; layer < num_layers_; ++layer) {
      Ort::Value &cache_key = (*cache_kv)[layer].first;
      Ort::Value &cache_val = (*cache_kv)[layer].second;

      const Ort::Value &delta_key = kv_delta[layer].first;
      const Ort::Value &delta_val = kv_delta[layer].second;

      auto ck_info = cache_key.GetTensorTypeAndShapeInfo();
      auto ck_shape = ck_info.GetShape();
      auto dk_shape = delta_key.GetTensorTypeAndShapeInfo().GetShape();
      auto cv_shape = cache_val.GetTensorTypeAndShapeInfo().GetShape();
      auto dv_shape = delta_val.GetTensorTypeAndShapeInfo().GetShape();

      if (ck_shape.size() < 4 || cv_shape.size() < 4 || dk_shape.size() < 4 ||
          dv_shape.size() < 4) {
        SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: KV tensors must be >= 4-D");
        SHERPA_ONNX_EXIT(-1);
      }

      int64_t B = ck_shape[0];
      int64_t kv_h = ck_shape[2];
      int64_t hd = ck_shape[3];
      if (B <= 0 || kv_h <= 0 || hd <= 0) {
        SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: invalid cache key shape");
        SHERPA_ONNX_EXIT(-1);
      }

      if (dk_shape[0] != B || dv_shape[0] != B || dk_shape[2] != kv_h ||
          dv_shape[2] != kv_h || dk_shape[3] != hd || dv_shape[3] != hd) {
        SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: delta shape mismatch");
        SHERPA_ONNX_EXIT(-1);
      }

      auto ck_type =
          static_cast<ONNXTensorElementDataType>(ck_info.GetElementType());
      auto cv_type = static_cast<ONNXTensorElementDataType>(
          cache_val.GetTensorTypeAndShapeInfo().GetElementType());
      auto dk_type = static_cast<ONNXTensorElementDataType>(
          delta_key.GetTensorTypeAndShapeInfo().GetElementType());
      auto dv_type = static_cast<ONNXTensorElementDataType>(
          delta_val.GetTensorTypeAndShapeInfo().GetElementType());

      if (ck_type != dk_type) {
        SHERPA_ONNX_LOGE(
            "ApplyKvDeltaInplace: key elem_type mismatch: cache=%d delta=%d",
            static_cast<int>(ck_type), static_cast<int>(dk_type));
        SHERPA_ONNX_EXIT(-1);
      }

      if (cv_type != dv_type) {
        SHERPA_ONNX_LOGE(
            "ApplyKvDeltaInplace: value elem_type mismatch: cache=%d delta=%d",
            static_cast<int>(cv_type), static_cast<int>(dv_type));
        SHERPA_ONNX_EXIT(-1);
      }

      if (dk_shape[1] != dv_shape[1]) {
        SHERPA_ONNX_LOGE(
            "ApplyKvDeltaInplace: delta key/value seq len mismatch: %d vs "
            "%d",
            static_cast<int32_t>(dk_shape[1]),
            static_cast<int32_t>(dv_shape[1]));
        SHERPA_ONNX_EXIT(-1);
      }

      size_t key_bytes_per_pos = static_cast<size_t>(kv_h) *
                                 static_cast<size_t>(hd) *
                                 ElemBytesFromTensorType(ck_type);
      size_t value_bytes_per_pos = static_cast<size_t>(kv_h) *
                                   static_cast<size_t>(hd) *
                                   ElemBytesFromTensorType(cv_type);

      void *dst_k = cache_key.GetTensorMutableData<void>();
      void *dst_v = cache_val.GetTensorMutableData<void>();
      const void *src_k = delta_key.GetTensorData<void>();
      const void *src_v = delta_val.GetTensorData<void>();

      int64_t copy_s = std::min<int64_t>(S, dk_shape[1]);
      if (copy_s <= 0) {
        continue;
      }

      for (int64_t b = 0; b < B; ++b) {
        size_t dst_k_off =
            (static_cast<size_t>(b) * static_cast<size_t>(max_total_len_) +
             static_cast<size_t>(pos0)) *
            key_bytes_per_pos;
        size_t src_k_off =
            (static_cast<size_t>(b) * static_cast<size_t>(dk_shape[1])) *
            key_bytes_per_pos;
        size_t copy_k_bytes = static_cast<size_t>(copy_s) * key_bytes_per_pos;

        size_t dst_v_off =
            (static_cast<size_t>(b) * static_cast<size_t>(max_total_len_) +
             static_cast<size_t>(pos0)) *
            value_bytes_per_pos;
        size_t src_v_off =
            (static_cast<size_t>(b) * static_cast<size_t>(dv_shape[1])) *
            value_bytes_per_pos;
        size_t copy_v_bytes = static_cast<size_t>(copy_s) * value_bytes_per_pos;

        uint8_t *dst_k_ptr = static_cast<uint8_t *>(dst_k) + dst_k_off;
        uint8_t *dst_v_ptr = static_cast<uint8_t *>(dst_v) + dst_v_off;
        const uint8_t *src_k_ptr =
            static_cast<const uint8_t *>(src_k) + src_k_off;
        const uint8_t *src_v_ptr =
            static_cast<const uint8_t *>(src_v) + src_v_off;

        std::memcpy(dst_k_ptr, src_k_ptr, copy_k_bytes);
        std::memcpy(dst_v_ptr, src_v_ptr, copy_v_bytes);
      }
    }
  }

  int32_t GetMaxTotalLen() const { return max_total_len_; }
  OrtAllocator *Allocator() { return allocator_; }

 private:
  OfflineModelConfig config_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_conv_;
  Ort::SessionOptions sess_opts_encoder_;
  Ort::SessionOptions sess_opts_decoder_;

  std::unique_ptr<Ort::Session> conv_sess_;
  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> conv_input_names_;
  std::vector<const char *> conv_input_names_ptr_;
  std::vector<std::string> conv_output_names_;
  std::vector<const char *> conv_output_names_ptr_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;
  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  ONNXTensorElementDataType kv_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType kv_in_type_v_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  Ort::AllocatorWithDefaultOptions allocator_;
  Ort::MemoryInfo cpu_mem_info_;
  std::unique_ptr<Ort::MemoryInfo> cuda_mem_info_;

  bool is_cpu_provider_ = true;
  bool use_cuda_iobinding_ = false;

  int32_t num_layers_ = 0;
  int32_t max_total_len_ = 0;

  std::vector<int64_t> past_key_shape_tpl_;
};

OfflineQwen3ASRModel::OfflineQwen3ASRModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineQwen3ASRModel::OfflineQwen3ASRModel(Manager *mgr,
                                           const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineQwen3ASRModel::~OfflineQwen3ASRModel() = default;

Ort::Value OfflineQwen3ASRModel::ForwardConvFrontend(
    Ort::Value input_features) {
  return impl_->ForwardConvFrontend(std::move(input_features));
}

Ort::Value OfflineQwen3ASRModel::ForwardEncoder(
    Ort::Value conv_output, Ort::Value feature_attention_mask) {
  return impl_->ForwardEncoder(std::move(conv_output),
                               std::move(feature_attention_mask));
}

std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
OfflineQwen3ASRModel::ForwardLLM(
    Ort::Value input_ids, Ort::Value audio_features, Ort::Value attention_mask,
    const Ort::Value &cache_position,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv) {
  return impl_->ForwardLLM(std::move(input_ids), std::move(audio_features),
                           std::move(attention_mask), cache_position, cache_kv);
}

std::vector<std::pair<Ort::Value, Ort::Value>>
OfflineQwen3ASRModel::CreateEmptyKVCache(int64_t batch) {
  return impl_->CreateEmptyKVCache(batch);
}

void OfflineQwen3ASRModel::ApplyKvDeltaInplace(
    std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
    const Ort::Value &cache_position) {
  impl_->ApplyKvDeltaInplace(cache_kv, kv_delta, cache_position);
}

int32_t OfflineQwen3ASRModel::GetMaxTotalLen() const {
  return impl_->GetMaxTotalLen();
}

OrtAllocator *OfflineQwen3ASRModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template OfflineQwen3ASRModel::OfflineQwen3ASRModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineQwen3ASRModel::OfflineQwen3ASRModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
