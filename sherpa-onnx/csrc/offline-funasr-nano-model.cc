// sherpa-onnx/csrc/offline-funasr-nano-model.cc
//
// Copyright (c)  2025  zengyw

#include "sherpa-onnx/csrc/offline-funasr-nano-model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
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

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Calculate the total number of elements from a tensor shape.
static inline size_t NumelFromShape(const std::vector<int64_t> &shape) {
  if (shape.empty()) return 0;
  size_t n = 1;
  for (auto d : shape) {
    if (d <= 0) return 0;
    n *= static_cast<size_t>(d);
  }
  return n;
}

static inline void AssertTensorIsCpu(const Ort::Value &v, const char *what) {
  if (!v.IsTensor()) return;
  auto mi = v.GetTensorMemoryInfo();
  if (mi.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    SHERPA_ONNX_LOGE(
        "%s: expected CPU tensor but got device_type=%d device_id=%d", what,
        (int)mi.GetDeviceType(), mi.GetDeviceId());
    SHERPA_ONNX_EXIT(-1);
  }
}

static inline std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) -> char {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

static inline bool IsCudaProvider(const std::string &provider) {
  auto p = ToLower(provider);
  // Keep it conservative. We only enable IO binding policy below when we
  // are on CUDA; other EPs keep the existing behavior.
  return p == "cuda" || (p.size() > 4 && p.find("cuda") == 0);
}

// Check if a tensor element type is FP16-IO (float16 or uint16).
static inline bool IsFloat16IO(ONNXTensorElementDataType t) {
  return t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
}

// Get the element type of a session input tensor.
static inline ONNXTensorElementDataType GetSessionInputElemType(
    Ort::Session *sess, size_t input_index) {
  auto ti = sess->GetInputTypeInfo(input_index);
  auto t = ti.GetTensorTypeAndShapeInfo();
  return static_cast<ONNXTensorElementDataType>(t.GetElementType());
}

// Get the element type of a session output tensor.
static inline ONNXTensorElementDataType GetSessionOutputElemType(
    Ort::Session *sess, size_t output_index) {
  auto ti = sess->GetOutputTypeInfo(output_index);
  auto t = ti.GetTensorTypeAndShapeInfo();
  return static_cast<ONNXTensorElementDataType>(t.GetElementType());
}

template <typename T>
static Ort::Value AllocTensor(OrtAllocator *alloc,
                              const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor<T>(alloc, shape.data(), shape.size());
}

template <>
Ort::Value AllocTensor<uint16_t>(OrtAllocator *alloc,
                                 const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor(alloc, shape.data(), shape.size(),
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
}

// Convert tensor to float32, handling both float16 and float32 inputs.
// NOTE: This helper assumes the input tensor is on CPU memory.
// The caller must ensure the tensor is on CPU (e.g., via IO Binding).
static Ort::Value CastToFloat32(Ort::Value in, OrtAllocator *alloc) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  size_t n = NumelFromShape(shape);
  if (n == 0) return in;
  auto et = info.GetElementType();

  AssertTensorIsCpu(in, "CastToFloat32");

  Ort::Value out = AllocTensor<float>(alloc, shape);
  float *dst = out.GetTensorMutableData<float>();
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *src = in.GetTensorData<float>();
    std::memcpy(dst, src, n * sizeof(float));
    return out;
  }
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
      et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    const uint16_t *src = in.GetTensorData<uint16_t>();
    for (size_t i = 0; i < n; ++i) dst[i] = HalfBitsToFloat(src[i]);
    return out;
  }
  SHERPA_ONNX_LOGE("CastToFloat32: unsupported input elem_type=%d", (int)et);
  return in;
}

// Convert tensor to float16, handling both float16 and float32 inputs.
// NOTE: This helper assumes the input tensor is on CPU memory.
static Ort::Value CastToFloat16(Ort::Value in, OrtAllocator *alloc) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  size_t n = NumelFromShape(shape);
  if (n == 0) return in;
  auto et = static_cast<ONNXTensorElementDataType>(info.GetElementType());

  AssertTensorIsCpu(in, "CastToFloat16");

  Ort::Value out = AllocTensor<uint16_t>(alloc, shape);
  uint16_t *dst = out.GetTensorMutableData<uint16_t>();
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
      et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    const uint16_t *src = in.GetTensorData<uint16_t>();
    std::memcpy(dst, src, n * sizeof(uint16_t));
    return out;
  }
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float *src = in.GetTensorData<float>();
    for (size_t i = 0; i < n; ++i) dst[i] = FloatToHalfBits(src[i]);
    return out;
  }
  SHERPA_ONNX_LOGE("CastToFloat16: unsupported input elem_type=%d", (int)et);
  return in;
}

// Cast tensor to the expected element type (float16 or float32).
// Returns the input unchanged if it already matches the expected type.
// NOTE: This helper assumes the input tensor is on CPU memory.
static Ort::Value CastFloatLikeForExpected(Ort::Value in,
                                           ONNXTensorElementDataType expected,
                                           OrtAllocator *alloc) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto actual = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (actual == expected) return in;
  if (expected == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return CastToFloat16(std::move(in), alloc);
  }
  if (expected == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return CastToFloat32(std::move(in), alloc);
  }
  SHERPA_ONNX_LOGE(
      "CastFloatLikeForExpected: unsupported expected elem_type=%d",
      (int)expected);
  return in;
}

static inline bool NeedsTypeConversion(Ort::Value &in,
                                       ONNXTensorElementDataType expected) {
  if (!in.IsTensor()) return false;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto actual = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  return actual != expected;
}

// Cast attention mask tensor to int64 if needed.
// Supports int32 to int64 conversion.
// NOTE: This helper assumes the input tensor is on CPU memory.
static Ort::Value CastMaskToInt64IfNeeded(Ort::Value in, OrtAllocator *alloc) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  size_t n = NumelFromShape(shape);
  if (n == 0) return in;
  auto et = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return in;

  AssertTensorIsCpu(in, "CastMaskToInt64IfNeeded");

  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t *src = in.GetTensorData<int32_t>();
    Ort::Value out = AllocTensor<int64_t>(alloc, shape);
    int64_t *dst = out.GetTensorMutableData<int64_t>();
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<int64_t>(src[i]);
    return out;
  }

  SHERPA_ONNX_LOGE("attention_mask elem_type=%d not supported, expected int64",
                   (int)et);
  return in;
}

// Create a non-owning tensor view that preserves the underlying memory info.
static inline Ort::Value ViewConst(const Ort::Value &v) {
  return View(const_cast<Ort::Value *>(&v));
}

}  // namespace

// Implementation class for OfflineFunASRNanoModel.
// Manages ONNX sessions for encoder, LLM prefill/decode, and embedding models.
class OfflineFunASRNanoModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "funasr-nano"),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_llm_(GetSessionOptions(config)),
        sess_opts_embedding_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.funasr_nano;

    if (c.encoder_adaptor.empty()) {
      SHERPA_ONNX_LOGE("funasr_nano.encoder_adaptor is empty");
      SHERPA_ONNX_EXIT(-1);
    }
    if (c.llm_prefill.empty() || c.llm_decode.empty()) {
      SHERPA_ONNX_LOGE(
          "funasr_nano.llm_prefill/llm_decode are required for KV-cache mode");
      SHERPA_ONNX_EXIT(-1);
    }

    InitEncoderAdaptor(c.encoder_adaptor);
    InitLLMPrefill(c.llm_prefill);
    InitLLMDecode(c.llm_decode);
    InitEmbedding(c.embedding);
    has_embedding_model_ = true;

    // FunASR-nano uses CPU-side sampling. When running on CUDA, we bind
    // logits to CPU (so sampling can read it safely), while keeping KV cache
    // on GPU to avoid large device<->host copies.
    use_cuda_iobinding_ =
        (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      // Use device 0 by default. SessionOptions() in sherpa-onnx usually
      // configures the CUDA EP device; binding here only affects output memory.
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

      // Check if prefill/decode models have FP16-IO, which is not supported on
      // CUDA yet.
      ONNXTensorElementDataType prefill_in_type = prefill_embeds_in_type_;
      ONNXTensorElementDataType prefill_out_type =
          GetSessionOutputElemType(prefill_sess_.get(), 0);
      ONNXTensorElementDataType decode_in_type = decode_embeds_in_type_;
      ONNXTensorElementDataType decode_out_type =
          GetSessionOutputElemType(decode_sess_.get(), 0);

      if (IsFloat16IO(prefill_in_type) || IsFloat16IO(prefill_out_type) ||
          IsFloat16IO(decode_in_type) || IsFloat16IO(decode_out_type)) {
        SHERPA_ONNX_LOGE(
            "fp16-IO LLM models are not supported on CUDA yet. Please use "
            "fp32/int8 models.");
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  void InitEncoderAdaptorFromMemory(void *model_data,
                                    size_t model_data_length) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_encoder_);
    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);
    encoder_in_type_ = GetSessionInputElemType(encoder_sess_.get(), 0);
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
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
    SHERPA_ONNX_READ_META_DATA(lfr_window_size_, "lfr_window_size");
    SHERPA_ONNX_READ_META_DATA(lfr_window_shift_, "lfr_window_shift");
    SHERPA_ONNX_READ_META_DATA(hidden_size_, "llm_dim");
  }

  void InitLLMPrefillFromMemory(void *model_data, size_t model_data_length) {
    prefill_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_llm_);
    GetInputNames(prefill_sess_.get(), &prefill_input_names_,
                  &prefill_input_names_ptr_);
    GetOutputNames(prefill_sess_.get(), &prefill_output_names_,
                   &prefill_output_names_ptr_);
    prefill_embeds_in_type_ = GetSessionInputElemType(prefill_sess_.get(), 0);
    Ort::ModelMetadata meta_data = prefill_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%{public}s\n",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%s\n", os.str().c_str());
#endif
    }
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }
  }

  void InitLLMDecodeFromMemory(void *model_data, size_t model_data_length) {
    decode_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_llm_);
    GetInputNames(decode_sess_.get(), &decode_input_names_,
                  &decode_input_names_ptr_);
    GetOutputNames(decode_sess_.get(), &decode_output_names_,
                   &decode_output_names_ptr_);
    decode_embeds_in_type_ = GetSessionInputElemType(decode_sess_.get(), 0);
    Ort::ModelMetadata meta_data = decode_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Decode model metadata:\n%{public}s\n",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("Decode model metadata:\n%s\n", os.str().c_str());
#endif
    }
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    int32_t decode_vocab_size = 0;
    SHERPA_ONNX_READ_META_DATA(decode_vocab_size, "vocab_size");
    if (vocab_size_ > 0 && decode_vocab_size != vocab_size_) {
      SHERPA_ONNX_LOGE(
          "Decode model vocab_size (%d) != prefill vocab_size (%d)",
          decode_vocab_size, vocab_size_);
    }
    if (vocab_size_ == 0) vocab_size_ = decode_vocab_size;
    int32_t decode_hidden_size = 0;
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(decode_hidden_size, "hidden_size");
      hidden_size_ = decode_hidden_size;
    }
  }

  void InitEmbeddingFromMemory(void *model_data, size_t model_data_length) {
    embedding_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_embedding_);
    GetInputNames(embedding_sess_.get(), &embedding_input_names_,
                  &embedding_input_names_ptr_);
    GetOutputNames(embedding_sess_.get(), &embedding_output_names_,
                   &embedding_output_names_ptr_);
    Ort::ModelMetadata meta_data = embedding_sess_->GetModelMetadata();
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
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "funasr-nano"),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_llm_(GetSessionOptions(config)),
        sess_opts_embedding_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.funasr_nano;

    if (c.encoder_adaptor.empty()) {
      SHERPA_ONNX_LOGE("funasr_nano.encoder_adaptor is empty");
      SHERPA_ONNX_EXIT(-1);
    }
    if (c.llm_prefill.empty() || c.llm_decode.empty()) {
      SHERPA_ONNX_LOGE(
          "funasr_nano.llm_prefill/llm_decode are required for KV-cache mode");
      SHERPA_ONNX_EXIT(-1);
    }

    auto buf_encoder = ReadFile(mgr, c.encoder_adaptor);
    InitEncoderAdaptorFromMemory(buf_encoder.data(), buf_encoder.size());

    {
      auto buf_prefill = ReadFile(mgr, c.llm_prefill);
      InitLLMPrefillFromMemory(buf_prefill.data(), buf_prefill.size());
    }

    auto buf_decode = ReadFile(mgr, c.llm_decode);
    InitLLMDecodeFromMemory(buf_decode.data(), buf_decode.size());

    auto buf_embedding = ReadFile(mgr, c.embedding);
    InitEmbeddingFromMemory(buf_embedding.data(), buf_embedding.size());
    has_embedding_model_ = true;

    use_cuda_iobinding_ =
        (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

      // Check if prefill/decode models have FP16-IO, which is not supported on
      // CUDA yet.
      ONNXTensorElementDataType prefill_in_type = prefill_embeds_in_type_;
      ONNXTensorElementDataType prefill_out_type =
          GetSessionOutputElemType(prefill_sess_.get(), 0);
      ONNXTensorElementDataType decode_in_type = decode_embeds_in_type_;
      ONNXTensorElementDataType decode_out_type =
          GetSessionOutputElemType(decode_sess_.get(), 0);

      if (IsFloat16IO(prefill_in_type) || IsFloat16IO(prefill_out_type) ||
          IsFloat16IO(decode_in_type) || IsFloat16IO(decode_out_type)) {
        SHERPA_ONNX_LOGE(
            "fp16-IO LLM models are not supported on CUDA yet. Please use "
            "fp32/int8 models.");
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  // Forward pass through encoder adaptor model.
  // Converts audio features to embeddings compatible with the LLM.
  Ort::Value ForwardEncoderAdaptor(Ort::Value features) {
    if (NeedsTypeConversion(features, encoder_in_type_)) {
      features = CastFloatLikeForExpected(std::move(features), encoder_in_type_,
                                          allocator_);
    }

    // Encoder output is consumed by CPU-side code (embedding packing), so we
    // bind it to CPU when running on CUDA to avoid returning a CUDA pointer.
    if (use_cuda_iobinding_) {
      Ort::IoBinding binding(*encoder_sess_);
      binding.BindInput(encoder_input_names_ptr_[0], features);
      binding.BindOutput(encoder_output_names_ptr_[0], cpu_mem_info_);
      binding.SynchronizeInputs();
      encoder_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      auto outs = binding.GetOutputValues();

      if (outs.empty()) {
        SHERPA_ONNX_LOGE("ForwardEncoderAdaptor: empty outputs");
        SHERPA_ONNX_EXIT(-1);
      }
      return std::move(outs[0]);
    }

    std::array<Ort::Value, 1> inputs = {std::move(features)};
    auto outputs = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    return std::move(outputs[0]);
  }

  // Forward pass through LLM prefill model with full context.
  // Returns logits and initial KV cache states for all layers.
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLMPrefill(Ort::Value inputs_embeds, Ort::Value attention_mask) {
    if (NeedsTypeConversion(inputs_embeds, prefill_embeds_in_type_)) {
      inputs_embeds = CastFloatLikeForExpected(
          std::move(inputs_embeds), prefill_embeds_in_type_, allocator_);
    }
    if (attention_mask.IsTensor()) {
      auto mask_info = attention_mask.GetTensorTypeAndShapeInfo();
      auto mask_type =
          static_cast<ONNXTensorElementDataType>(mask_info.GetElementType());
      if (mask_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        attention_mask =
            CastMaskToInt64IfNeeded(std::move(attention_mask), allocator_);
      }
    }

    std::vector<Ort::Value> outputs;

    if (use_cuda_iobinding_) {
      // CPU-side sampling needs logits on CPU, while KV cache should remain on
      // GPU to avoid large device<->host copies between decode steps.
      Ort::IoBinding binding(*prefill_sess_);
      binding.BindInput(prefill_input_names_ptr_[0], inputs_embeds);
      binding.BindInput(prefill_input_names_ptr_[1], attention_mask);

      binding.BindOutput(prefill_output_names_ptr_[0], cpu_mem_info_);
      for (size_t i = 1; i < prefill_output_names_ptr_.size(); ++i) {
        binding.BindOutput(prefill_output_names_ptr_[i], *cuda_mem_info_);
      }

      binding.SynchronizeInputs();
      prefill_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      outputs = binding.GetOutputValues();
    } else {
      std::array<Ort::Value, 2> inputs = {std::move(inputs_embeds),
                                          std::move(attention_mask)};
      outputs = prefill_sess_->Run(
          {}, prefill_input_names_ptr_.data(), inputs.data(), inputs.size(),
          prefill_output_names_ptr_.data(), prefill_output_names_ptr_.size());
    }

    // First output is logits, remaining outputs are past_key_values
    if (outputs.empty()) {
      SHERPA_ONNX_LOGE("ForwardLLMPrefill: empty outputs");
      SHERPA_ONNX_EXIT(-1);
    }

    Ort::Value logits = std::move(outputs[0]);

    if (!logits.IsTensor()) {
      SHERPA_ONNX_LOGE("ForwardLLMPrefill: logits is not a tensor");
      SHERPA_ONNX_EXIT(-1);
    }
    AssertTensorIsCpu(logits, "ForwardLLMPrefill logits");

    if ((outputs.size() - 1) % 2 != 0) {
      SHERPA_ONNX_LOGE("ForwardLLMPrefill: invalid KV cache outputs size=%d",
                       static_cast<int>(outputs.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<std::pair<Ort::Value, Ort::Value>> past_kv;
    int num_layers = static_cast<int>((outputs.size() - 1) / 2);
    past_kv.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      past_kv.emplace_back(std::move(outputs[1 + 2 * i]),
                           std::move(outputs[1 + 2 * i + 1]));
    }
    return {std::move(logits), std::move(past_kv)};
  }

  // Forward pass through LLM decode model with KV cache.
  // Takes a single token embedding and past KV cache, returns logits and
  // updated KV cache states.
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLMDecode(
      Ort::Value inputs_embeds, Ort::Value attention_mask,
      const std::vector<std::pair<Ort::Value, Ort::Value>> &past_key_values) {
    if (NeedsTypeConversion(inputs_embeds, decode_embeds_in_type_)) {
      inputs_embeds = CastFloatLikeForExpected(
          std::move(inputs_embeds), decode_embeds_in_type_, allocator_);
    }
    if (attention_mask.IsTensor()) {
      auto mask_info = attention_mask.GetTensorTypeAndShapeInfo();
      auto mask_type =
          static_cast<ONNXTensorElementDataType>(mask_info.GetElementType());
      if (mask_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        attention_mask =
            CastMaskToInt64IfNeeded(std::move(attention_mask), allocator_);
      }
    }

    // Build inputs: [inputs_embeds, attention_mask, past_key_0, past_value_0,
    // ...] NOTE: We create non-owning Ort::Value views that reference existing
    // buffers.
    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + 2 * past_key_values.size());
    inputs.push_back(std::move(inputs_embeds));
    inputs.push_back(std::move(attention_mask));
    for (const auto &kv : past_key_values) {
      inputs.push_back(ViewConst(kv.first));
      inputs.push_back(ViewConst(kv.second));
    }

    // Build input names: [inputs_embeds, attention_mask, past_key_0,
    // past_value_0, ...]
    std::vector<const char *> input_names_ptr;
    input_names_ptr.reserve(2 + 2 * past_key_values.size());
    input_names_ptr.push_back(decode_input_names_ptr_[0]);
    input_names_ptr.push_back(decode_input_names_ptr_[1]);
    for (size_t i = 0; i < past_key_values.size(); ++i) {
      input_names_ptr.push_back(decode_input_names_ptr_[2 + 2 * i]);
      input_names_ptr.push_back(decode_input_names_ptr_[2 + 2 * i + 1]);
    }

    std::vector<Ort::Value> outputs;

    if (use_cuda_iobinding_) {
      // Bind logits to CPU for sampling; keep KV cache on GPU for next steps.
      Ort::IoBinding binding(*decode_sess_);
      for (size_t i = 0; i < inputs.size(); ++i) {
        binding.BindInput(input_names_ptr[i], inputs[i]);
      }

      binding.BindOutput(decode_output_names_ptr_[0], cpu_mem_info_);
      for (size_t i = 1; i < decode_output_names_ptr_.size(); ++i) {
        binding.BindOutput(decode_output_names_ptr_[i], *cuda_mem_info_);
      }

      binding.SynchronizeInputs();
      decode_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      outputs = binding.GetOutputValues();
    } else {
      outputs = decode_sess_->Run(
          {}, input_names_ptr.data(), inputs.data(), inputs.size(),
          decode_output_names_ptr_.data(), decode_output_names_ptr_.size());
    }

    // First output is logits, remaining outputs are updated past_key_values
    if (outputs.empty()) {
      SHERPA_ONNX_LOGE("ForwardLLMDecode: empty outputs");
      SHERPA_ONNX_EXIT(-1);
    }

    Ort::Value logits = std::move(outputs[0]);

    if (!logits.IsTensor()) {
      SHERPA_ONNX_LOGE("ForwardLLMDecode: logits is not a tensor");
      SHERPA_ONNX_EXIT(-1);
    }
    AssertTensorIsCpu(logits, "ForwardLLMDecode logits");

    if ((outputs.size() - 1) % 2 != 0) {
      SHERPA_ONNX_LOGE("ForwardLLMDecode: invalid KV cache outputs size=%d",
                       static_cast<int>(outputs.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<std::pair<Ort::Value, Ort::Value>> updated_past_kv;
    int num_layers = static_cast<int>((outputs.size() - 1) / 2);
    updated_past_kv.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
      updated_past_kv.emplace_back(std::move(outputs[1 + 2 * i]),
                                   std::move(outputs[1 + 2 * i + 1]));
    }
    return {std::move(logits), std::move(updated_past_kv)};
  }

  // Forward pass through embedding model.
  // Converts token IDs to embeddings.
  Ort::Value ForwardEmbedding(Ort::Value input_ids) {
    // Embedding output is consumed by CPU-side packing code; bind it to CPU
    // when running on CUDA to avoid returning a CUDA pointer.
    if (use_cuda_iobinding_) {
      Ort::IoBinding binding(*embedding_sess_);
      binding.BindInput(embedding_input_names_ptr_[0], input_ids);
      binding.BindOutput(embedding_output_names_ptr_[0], cpu_mem_info_);
      binding.SynchronizeInputs();
      embedding_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      auto outs = binding.GetOutputValues();

      if (outs.empty()) {
        SHERPA_ONNX_LOGE("ForwardEmbedding: empty outputs");
        SHERPA_ONNX_EXIT(-1);
      }
      return std::move(outs[0]);
    }

    std::array<Ort::Value, 1> inputs = {std::move(input_ids)};
    auto outputs = embedding_sess_->Run(
        {}, embedding_input_names_ptr_.data(), inputs.data(), inputs.size(),
        embedding_output_names_ptr_.data(), embedding_output_names_ptr_.size());
    return std::move(outputs[0]);
  }

  int32_t VocabSize() const { return vocab_size_; }
  int32_t HiddenSize() const { return hidden_size_; }
  int32_t LfrWindowSize() const { return lfr_window_size_; }
  int32_t LfrWindowShift() const { return lfr_window_shift_; }
  OrtAllocator *Allocator() { return allocator_; }
  bool HasEmbeddingModel() const { return has_embedding_model_; }
  bool UseKVCache() const { return true; }
  ONNXTensorElementDataType GetPrefillInputType() const {
    return prefill_embeds_in_type_;
  }
  ONNXTensorElementDataType GetDecodeInputType() const {
    return decode_embeds_in_type_;
  }
  bool IsCpuProvider() const { return is_cpu_provider_; }

 private:
  void InitEncoderAdaptor(const std::string &model_path) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(model_path), sess_opts_encoder_);
    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);
    encoder_in_type_ = GetSessionInputElemType(encoder_sess_.get(), 0);
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
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
    SHERPA_ONNX_READ_META_DATA(lfr_window_size_, "lfr_window_size");
    SHERPA_ONNX_READ_META_DATA(lfr_window_shift_, "lfr_window_shift");
    SHERPA_ONNX_READ_META_DATA(hidden_size_, "llm_dim");
  }

  void InitLLMPrefill(const std::string &model_path) {
    prefill_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(model_path), sess_opts_llm_);
    GetInputNames(prefill_sess_.get(), &prefill_input_names_,
                  &prefill_input_names_ptr_);
    GetOutputNames(prefill_sess_.get(), &prefill_output_names_,
                   &prefill_output_names_ptr_);
    prefill_embeds_in_type_ = GetSessionInputElemType(prefill_sess_.get(), 0);
    Ort::ModelMetadata meta_data = prefill_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%{public}s\n",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%s\n", os.str().c_str());
#endif
    }
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }
  }

  void InitLLMDecode(const std::string &model_path) {
    decode_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(model_path), sess_opts_llm_);
    GetInputNames(decode_sess_.get(), &decode_input_names_,
                  &decode_input_names_ptr_);
    GetOutputNames(decode_sess_.get(), &decode_output_names_,
                   &decode_output_names_ptr_);
    decode_embeds_in_type_ = GetSessionInputElemType(decode_sess_.get(), 0);
    Ort::ModelMetadata meta_data = decode_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Decode model metadata:\n%{public}s\n",
                       os.str().c_str());
#else
      SHERPA_ONNX_LOGE("Decode model metadata:\n%s\n", os.str().c_str());
#endif
    }
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    int32_t decode_vocab_size = 0;
    SHERPA_ONNX_READ_META_DATA(decode_vocab_size, "vocab_size");
    if (vocab_size_ > 0 && decode_vocab_size != vocab_size_) {
      SHERPA_ONNX_LOGE(
          "Decode model vocab_size (%d) != prefill vocab_size (%d)",
          decode_vocab_size, vocab_size_);
    }
    if (vocab_size_ == 0) vocab_size_ = decode_vocab_size;
    int32_t decode_hidden_size = 0;
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(decode_hidden_size, "hidden_size");
      hidden_size_ = decode_hidden_size;
    }
  }

  void InitEmbedding(const std::string &model_path) {
    embedding_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(model_path), sess_opts_embedding_);
    GetInputNames(embedding_sess_.get(), &embedding_input_names_,
                  &embedding_input_names_ptr_);
    GetOutputNames(embedding_sess_.get(), &embedding_output_names_,
                   &embedding_output_names_ptr_);
    Ort::ModelMetadata meta_data = embedding_sess_->GetModelMetadata();
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
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_encoder_;
  Ort::SessionOptions sess_opts_llm_;
  Ort::SessionOptions sess_opts_embedding_;
  Ort::AllocatorWithDefaultOptions allocator_;

  Ort::MemoryInfo cpu_mem_info_;
  std::unique_ptr<Ort::MemoryInfo> cuda_mem_info_;
  bool use_cuda_iobinding_ = false;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> prefill_sess_;
  std::unique_ptr<Ort::Session> decode_sess_;
  std::unique_ptr<Ort::Session> embedding_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> prefill_input_names_;
  std::vector<const char *> prefill_input_names_ptr_;
  std::vector<std::string> prefill_output_names_;
  std::vector<const char *> prefill_output_names_ptr_;

  std::vector<std::string> decode_input_names_;
  std::vector<const char *> decode_input_names_ptr_;
  std::vector<std::string> decode_output_names_;
  std::vector<const char *> decode_output_names_ptr_;

  std::vector<std::string> embedding_input_names_;
  std::vector<const char *> embedding_input_names_ptr_;
  std::vector<std::string> embedding_output_names_;
  std::vector<const char *> embedding_output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t hidden_size_ = 0;
  int32_t lfr_window_size_ = 0;
  int32_t lfr_window_shift_ = 0;

  bool has_embedding_model_ = false;

  ONNXTensorElementDataType encoder_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType prefill_embeds_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType decode_embeds_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  bool is_cpu_provider_ = false;
};

OfflineFunASRNanoModel::OfflineFunASRNanoModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineFunASRNanoModel::OfflineFunASRNanoModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineFunASRNanoModel::~OfflineFunASRNanoModel() = default;

Ort::Value OfflineFunASRNanoModel::ForwardEncoderAdaptor(Ort::Value features) {
  return impl_->ForwardEncoderAdaptor(std::move(features));
}

std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
OfflineFunASRNanoModel::ForwardLLMPrefill(Ort::Value inputs_embeds,
                                          Ort::Value attention_mask) {
  return impl_->ForwardLLMPrefill(std::move(inputs_embeds),
                                  std::move(attention_mask));
}

std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
OfflineFunASRNanoModel::ForwardLLMDecode(
    Ort::Value inputs_embeds, Ort::Value attention_mask,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &past_key_values) {
  return impl_->ForwardLLMDecode(std::move(inputs_embeds),
                                 std::move(attention_mask), past_key_values);
}

bool OfflineFunASRNanoModel::UseKVCache() const { return impl_->UseKVCache(); }

Ort::Value OfflineFunASRNanoModel::ForwardEmbedding(Ort::Value input_ids) {
  return impl_->ForwardEmbedding(std::move(input_ids));
}

int32_t OfflineFunASRNanoModel::VocabSize() const { return impl_->VocabSize(); }
int32_t OfflineFunASRNanoModel::HiddenSize() const {
  return impl_->HiddenSize();
}
int32_t OfflineFunASRNanoModel::LfrWindowSize() const {
  return impl_->LfrWindowSize();
}
int32_t OfflineFunASRNanoModel::LfrWindowShift() const {
  return impl_->LfrWindowShift();
}

OrtAllocator *OfflineFunASRNanoModel::Allocator() const {
  return impl_->Allocator();
}

bool OfflineFunASRNanoModel::HasEmbeddingModel() const {
  return impl_->HasEmbeddingModel();
}

ONNXTensorElementDataType OfflineFunASRNanoModel::GetPrefillInputType() const {
  return impl_->GetPrefillInputType();
}

ONNXTensorElementDataType OfflineFunASRNanoModel::GetDecodeInputType() const {
  return impl_->GetDecodeInputType();
}

#if __ANDROID_API__ >= 9
template OfflineFunASRNanoModel::OfflineFunASRNanoModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineFunASRNanoModel::OfflineFunASRNanoModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
