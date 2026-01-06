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
#include <climits>

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

#if ORT_API_VERSION >= 14
static inline void AssertTensorIsCpu(const Ort::Value &v, const char *what) {
  if (!v.IsTensor()) return;
  auto mi = v.GetTensorMemoryInfo();
  if (mi.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
    SHERPA_ONNX_LOGE("%s: expected CPU tensor but got device_type=%d device_id=%d",
                     what, (int)mi.GetDeviceType(), mi.GetDeviceId());
    SHERPA_ONNX_EXIT(-1);
  }
}
#else
static inline void AssertTensorIsCpu(const Ort::Value &v, const char *what) {
  if (!v.IsTensor()) return;

  const OrtValue* v_ptr = reinterpret_cast<const OrtValue*>(&v);
  const OrtMemoryInfo* memory_info = nullptr;

  // 1. Get memory info
  OrtStatus* status = Ort::GetApi().GetTensorMemoryInfo(v_ptr, &memory_info);
  if (status) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    SHERPA_ONNX_LOGE("%s: failed to get tensor memory info: %s", what, msg);
    SHERPA_ONNX_EXIT(-1);
  }

  // 2. Get memory type (OrtMemType)
  OrtMemType mem_type;
  status = Ort::GetApi().MemoryInfoGetMemType(memory_info, &mem_type);
  if (status) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    SHERPA_ONNX_LOGE("%s: failed to get mem type: %s", what, msg);
    SHERPA_ONNX_EXIT(-1);
  }

  // 3. Check CPU
  if (mem_type != OrtMemTypeCPU) {
    int device_id = 0;
    status = Ort::GetApi().MemoryInfoGetId(memory_info, &device_id);
    if (status) {
      const char* msg = Ort::GetApi().GetErrorMessage(status);
      Ort::GetApi().ReleaseStatus(status);
      SHERPA_ONNX_LOGE("%s: failed to get device id: %s", what, msg);
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_LOGE("%s: expected CPU tensor but got mem_type=%d device_id=%d",
                     what, static_cast<int>(mem_type), device_id);
    SHERPA_ONNX_EXIT(-1);
  }
}
#endif

static inline std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) -> char {
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

// Allocate tensor by ONNX elem type (float/float16 only).
static inline Ort::Value AllocTensorByElemType(OrtAllocator *alloc,
                                               const std::vector<int64_t> &shape,
                                               ONNXTensorElementDataType t) {
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return AllocTensor<float>(alloc, shape);
  }
  if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
      t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    return AllocTensor<uint16_t>(alloc, shape);
  }
  SHERPA_ONNX_LOGE("AllocTensorByElemType: unsupported elem_type=%d", (int)t);
  SHERPA_ONNX_EXIT(-1);
  return AllocTensor<float>(alloc, shape);
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

// Ensure attention_mask is [batch, target_len] on CPU, int64.
// If shorter: pad with 0. If longer: truncate.
static Ort::Value NormalizeAttentionMask(Ort::Value mask, int64_t target_len,
                                        OrtAllocator *alloc) {
  if (!mask.IsTensor()) return mask;
  AssertTensorIsCpu(mask, "NormalizeAttentionMask");

  auto info = mask.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 2) return mask;

  int64_t b = shape[0];
  int64_t l = shape[1];
  if (b <= 0 || l <= 0) return mask;

  if (static_cast<ONNXTensorElementDataType>(info.GetElementType()) !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    mask = CastMaskToInt64IfNeeded(std::move(mask), alloc);
    info = mask.GetTensorTypeAndShapeInfo();
    shape = info.GetShape();
    if (shape.size() != 2) return mask;
    b = shape[0];
    l = shape[1];
  }

  if (l == target_len) return mask;

  std::vector<int64_t> new_shape = {b, target_len};
  Ort::Value out = AllocTensor<int64_t>(alloc, new_shape);
  int64_t *dst = out.GetTensorMutableData<int64_t>();
  const int64_t *src = mask.GetTensorData<int64_t>();

  std::memset(dst, 0, static_cast<size_t>(b) * static_cast<size_t>(target_len) *
                          sizeof(int64_t));

  int64_t copy_len = std::min<int64_t>(l, target_len);
  for (int64_t bi = 0; bi < b; ++bi) {
    const int64_t *srow = src + bi * l;
    int64_t *drow = dst + bi * target_len;
    std::memcpy(drow, srow, static_cast<size_t>(copy_len) * sizeof(int64_t));
  }

  return out;
}

}  // namespace

// Implementation class for OfflineFunASRNanoModel.
// Manages ONNX sessions for encoder, KV cache LLM, and embedding models.
class OfflineFunASRNanoModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "funasr-nano"),
        sess_opts_encoder_(GetSessionOptions(config)),
        sess_opts_llm_(GetSessionOptions(config)),
        sess_opts_embedding_(GetSessionOptions(config)),
        allocator_(),
        cpu_mem_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                                 OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.funasr_nano;

    if (c.encoder_adaptor.empty()) {
      SHERPA_ONNX_LOGE("funasr_nano.encoder_adaptor is empty");
      SHERPA_ONNX_EXIT(-1);
    }

    if (c.llm.empty()) {
      SHERPA_ONNX_LOGE("funasr_nano.llm is required for KV cache mode");
      SHERPA_ONNX_EXIT(-1);
    }
    
    InitEncoderAdaptor(c.encoder_adaptor);
    InitLLM(c.llm);
    InitEmbedding(c.embedding);
    has_embedding_model_ = true;

    // FunASR-nano uses CPU-side sampling. When running on CUDA, we bind
    // logits to CPU (so sampling can read it safely).
    use_cuda_iobinding_ = (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      // Use device 0 by default. SessionOptions() in sherpa-onnx usually
      // configures the CUDA EP device; binding here only affects output memory.
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
    CheckFp16OnCuda();
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

  void InitLLMFromMemory(void *model_data, size_t model_data_length) {
    try {
      llm_sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                                 model_data_length,
                                                 sess_opts_llm_);
    } catch (const Ort::Exception &e) {
      SHERPA_ONNX_LOGE("InitLLMFromMemory: failed to create session: %s", e.what());
      if (std::string(e.what()).find("external data") != std::string::npos ||
          std::string(e.what()).find("External data") != std::string::npos) {
        SHERPA_ONNX_LOGE(
            "LLM model requires external data (.data file) but loaded from memory. "
            "Please use fp16/int8 single-file model or load by file path instead.");
        SHERPA_ONNX_EXIT(-1);
      }
      throw;
    }

    GetInputNames(llm_sess_.get(), &llm_input_names_, &llm_input_names_ptr_);
    GetOutputNames(llm_sess_.get(), &llm_output_names_, &llm_output_names_ptr_);

    llm_embeds_in_type_ = GetSessionInputElemType(llm_sess_.get(), 0);
    if (llm_embeds_in_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      SHERPA_ONNX_LOGE("LLM inputs_embeds must be float32, got elem_type=%d",
                       (int)llm_embeds_in_type_);
      SHERPA_ONNX_EXIT(-1);
    }

    Ort::ModelMetadata meta_data = llm_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("LLM model metadata:\n%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("LLM model metadata:\n%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }

    // Detect KV delta model type (model_type metadata should contain "kv_delta")
    auto model_type_value =
        LookupCustomModelMetaData(meta_data, "model_type", allocator);
    is_kv_delta_model_ = (!model_type_value.empty() &&
                          model_type_value.find("kv_delta") != std::string::npos);

    int32_t num_outputs = static_cast<int32_t>(llm_output_names_.size());
    if (num_outputs < 1 || (num_outputs - 1) % 2 != 0) {
      SHERPA_ONNX_LOGE(
          "LLM model must have 1 logits output + 2*num_layers KV outputs, got %d outputs",
          num_outputs);
      SHERPA_ONNX_EXIT(-1);
    }
    int32_t inferred_layers = (num_outputs - 1) / 2;

    auto num_layers_value =
        LookupCustomModelMetaData(meta_data, "num_layers", allocator);
    if (!num_layers_value.empty()) {
      num_layers_ = atoi(num_layers_value.c_str());
      if (num_layers_ <= 0) {
        SHERPA_ONNX_LOGE("Invalid num_layers=%d from metadata", num_layers_);
        SHERPA_ONNX_EXIT(-1);
      }
      if (num_layers_ != inferred_layers) {
        SHERPA_ONNX_LOGE("LLM num_layers mismatch: metadata=%d, inferred=%d",
                         num_layers_, inferred_layers);
        SHERPA_ONNX_EXIT(-1);
      }
    } else {
      num_layers_ = inferred_layers;
    }

    // Read KV cache capacity from metadata.
    auto max_total_len_value =
        LookupCustomModelMetaData(meta_data, "max_total_len", allocator);
    if (!max_total_len_value.empty()) {
      max_total_len_ = atoi(max_total_len_value.c_str());
    } else {
      auto attn_len_value =
          LookupCustomModelMetaData(meta_data, "attention_mask_len", allocator);
      if (!attn_len_value.empty()) max_total_len_ = atoi(attn_len_value.c_str());
    }
    if (max_total_len_ <= 0) {
      // Fallback: use input[1] shape
      auto ti = llm_sess_->GetInputTypeInfo(1);
      auto shp = ti.GetTensorTypeAndShapeInfo().GetShape();
      if (shp.size() == 2 && shp[1] > 0) {
        max_total_len_ = static_cast<int32_t>(shp[1]);
      }
      if (max_total_len_ <= 0) {
        SHERPA_ONNX_LOGE("Failed to determine max_total_len from metadata or input shape");
        SHERPA_ONNX_EXIT(-1);
      }
    }

    // Only KV delta models are supported
    if (!is_kv_delta_model_) {
      SHERPA_ONNX_LOGE("Only KV delta models are supported, but model_type does not contain 'kv_delta'");
      SHERPA_ONNX_EXIT(-1);
    }

    // Validate input layout: 0 embeds, 1 attention_mask, 2 cache_position, 3+ KV cache
    if (llm_input_names_.size() < 3u) {
      SHERPA_ONNX_LOGE("LLM model inputs must be >=3 (embeds,mask,cache_position)");
      SHERPA_ONNX_EXIT(-1);
    }

    cache_position_input_index_ = 2;
    past_kv_input_start_index_ = 3;

    int32_t expected_inputs = 3 + 2 * num_layers_;
    int32_t actual_inputs = static_cast<int32_t>(llm_input_names_.size());
    if (actual_inputs != expected_inputs) {
      if (actual_inputs == 2 + 2 * num_layers_) {
        SHERPA_ONNX_LOGE(
            "LLM model inputs mismatch: expected %d (=3+2*num_layers with cache_position) "
            "got %d (=2+2*num_layers without cache_position). "
            "Please use a model exported with cache_position support.",
            expected_inputs, actual_inputs);
      } else {
        SHERPA_ONNX_LOGE(
            "LLM model inputs mismatch: expected %d (=3+2*num_layers) got %d",
            expected_inputs, actual_inputs);
      }
      SHERPA_ONNX_EXIT(-1);
    }

    // KV input element type (should be float16 or float32).
    kv_in_type_ = GetSessionInputElemType(llm_sess_.get(), past_kv_input_start_index_);
    kv_in_type_v_ = GetSessionInputElemType(llm_sess_.get(), past_kv_input_start_index_ + 1);
    if (!(kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
      SHERPA_ONNX_LOGE("LLM past_key elem_type=%d not supported", (int)kv_in_type_);
      SHERPA_ONNX_EXIT(-1);
    }
    if (!(kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
      SHERPA_ONNX_LOGE("LLM past_value elem_type=%d not supported", (int)kv_in_type_v_);
      SHERPA_ONNX_EXIT(-1);
    }

    // Templates for KV shapes from session inputs.
    auto past_key_ti = llm_sess_->GetInputTypeInfo(past_kv_input_start_index_);
    past_key_shape_tpl_ = past_key_ti.GetTensorTypeAndShapeInfo().GetShape();

    auto past_value_ti = llm_sess_->GetInputTypeInfo(past_kv_input_start_index_ + 1);
    past_value_shape_tpl_ = past_value_ti.GetTensorTypeAndShapeInfo().GetShape();
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
        cpu_mem_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                                 OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.funasr_nano;

    if (c.encoder_adaptor.empty()) {
      SHERPA_ONNX_LOGE("funasr_nano.encoder_adaptor is empty");
      SHERPA_ONNX_EXIT(-1);
    }

    if (c.llm.empty()) {
      SHERPA_ONNX_LOGE("funasr_nano.llm is required for KV cache mode");
      SHERPA_ONNX_EXIT(-1);
    }

    auto buf_encoder = ReadFile(mgr, c.encoder_adaptor);
    InitEncoderAdaptorFromMemory(buf_encoder.data(), buf_encoder.size());

    auto buf_llm = ReadFile(mgr, c.llm);
    InitLLMFromMemory(buf_llm.data(), buf_llm.size());

    auto buf_embedding = ReadFile(mgr, c.embedding);
    InitEmbeddingFromMemory(buf_embedding.data(), buf_embedding.size());
    has_embedding_model_ = true;

    use_cuda_iobinding_ = (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
    CheckFp16OnCuda();
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

  std::vector<std::pair<Ort::Value, Ort::Value>> CreateEmptyKVCache(int64_t batch) {
    std::vector<std::pair<Ort::Value, Ort::Value>> kv_cache;
    kv_cache.reserve(num_layers_);

    // Read kv_h, hd from input shape template (dim2, dim3)
    auto &tpl = past_key_shape_tpl_;
    if (tpl.size() < 4) {
      SHERPA_ONNX_LOGE("Invalid KV cache shape template, expected >=4 dims");
      SHERPA_ONNX_EXIT(-1);
    }
    int64_t kv_h = tpl[2];
    int64_t hd = tpl[3];
    std::vector<int64_t> key_shape = {batch, static_cast<int64_t>(max_total_len_), kv_h, hd};
    std::vector<int64_t> value_shape = key_shape;

    size_t key_numel = NumelFromShape(key_shape);
    size_t value_numel = NumelFromShape(value_shape);

    for (int32_t i = 0; i < num_layers_; ++i) {
      Ort::Value key_tensor =
          AllocTensorByElemType(allocator_, key_shape, kv_in_type_);
      Ort::Value value_tensor =
          AllocTensorByElemType(allocator_, value_shape, kv_in_type_);

      // Zero-initialize cache
      if (key_numel > 0) {
        if (kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          std::memset(key_tensor.GetTensorMutableData<float>(), 0, key_numel * sizeof(float));
        } else {
          std::memset(key_tensor.GetTensorMutableData<uint16_t>(), 0, key_numel * sizeof(uint16_t));
        }
      }

      if (value_numel > 0) {
        if (kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          std::memset(value_tensor.GetTensorMutableData<float>(), 0, value_numel * sizeof(float));
        } else {
          std::memset(value_tensor.GetTensorMutableData<uint16_t>(), 0, value_numel * sizeof(uint16_t));
        }
      }

      kv_cache.emplace_back(std::move(key_tensor), std::move(value_tensor));
    }
    return kv_cache;
  }

  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>> ForwardLLM(
      Ort::Value inputs_embeds, Ort::Value attention_mask,
      const Ort::Value &cache_position,
      const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv) {
    if (static_cast<int32_t>(cache_kv.size()) != num_layers_) {
      SHERPA_ONNX_LOGE("ForwardLLM: cache_kv size (%zu) != num_layers (%d)",
                       cache_kv.size(), num_layers_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (!inputs_embeds.IsTensor()) {
      SHERPA_ONNX_LOGE("ForwardLLM: inputs_embeds is not a tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    auto embeds_info = inputs_embeds.GetTensorTypeAndShapeInfo();
    auto embeds_type =
        static_cast<ONNXTensorElementDataType>(embeds_info.GetElementType());
    if (embeds_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      SHERPA_ONNX_LOGE("ForwardLLM: inputs_embeds must be float32, got elem_type=%d",
                       (int)embeds_type);
      SHERPA_ONNX_EXIT(-1);
    }

    // Prepare attention_mask: int64, ensure length <= max_total_len
    if (attention_mask.IsTensor()) {
      auto mask_info = attention_mask.GetTensorTypeAndShapeInfo();
      auto mask_type =
          static_cast<ONNXTensorElementDataType>(mask_info.GetElementType());
      if (mask_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        attention_mask = CastMaskToInt64IfNeeded(std::move(attention_mask), allocator_);
        mask_info = attention_mask.GetTensorTypeAndShapeInfo();
      }

      auto mask_shape = mask_info.GetShape();
      if (mask_shape.size() == 2 && mask_shape[1] > max_total_len_) {
        // Truncate attention_mask if it exceeds max_total_len
        attention_mask = NormalizeAttentionMask(std::move(attention_mask),
                                                max_total_len_, allocator_);
      }
    }

    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + 2 * cache_kv.size());
    inputs.push_back(std::move(inputs_embeds));
    inputs.push_back(std::move(attention_mask));
    inputs.push_back(View(const_cast<Ort::Value *>(&cache_position)));

    for (const auto &kv : cache_kv) {
      inputs.push_back(View(const_cast<Ort::Value *>(&kv.first)));
      inputs.push_back(View(const_cast<Ort::Value *>(&kv.second)));
    }

    std::vector<const char *> input_names_ptr;
    input_names_ptr.reserve(3 + 2 * cache_kv.size());
    input_names_ptr.push_back(llm_input_names_ptr_[0]);  // inputs_embeds
    input_names_ptr.push_back(llm_input_names_ptr_[1]);  // attention_mask
    input_names_ptr.push_back(llm_input_names_ptr_[2]);  // cache_position
    for (size_t i = 0; i < cache_kv.size(); ++i) {
      input_names_ptr.push_back(llm_input_names_ptr_[past_kv_input_start_index_ + 2 * i]);
      input_names_ptr.push_back(llm_input_names_ptr_[past_kv_input_start_index_ + 2 * i + 1]);
    }

    std::vector<Ort::Value> outputs;

    if (use_cuda_iobinding_) {
      Ort::IoBinding binding(*llm_sess_);
      for (size_t i = 0; i < inputs.size(); ++i) {
        binding.BindInput(input_names_ptr[i], inputs[i]);
      }

      // logits must be CPU (we will read it on CPU).
      binding.BindOutput(llm_output_names_ptr_[0], cpu_mem_info_);

      // KV outputs: bind to CPU so ApplyKvDeltaInplace can work with CPU cache
      for (size_t i = 1; i < llm_output_names_ptr_.size(); ++i) {
        binding.BindOutput(llm_output_names_ptr_[i], cpu_mem_info_);
      }

      binding.SynchronizeInputs();
      llm_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      outputs = binding.GetOutputValues();
    } else {
      outputs = llm_sess_->Run({}, input_names_ptr.data(), inputs.data(),
                               inputs.size(), llm_output_names_ptr_.data(),
                               llm_output_names_ptr_.size());
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

    AssertTensorIsCpu(logits, "ForwardLLM logits");

    auto logits_info = logits.GetTensorTypeAndShapeInfo();
    auto logits_type =
        static_cast<ONNXTensorElementDataType>(logits_info.GetElementType());
    if (logits_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      SHERPA_ONNX_LOGE("ForwardLLM: logits must be float32, got elem_type=%d",
                       (int)logits_type);
      SHERPA_ONNX_EXIT(-1);
    }

    if ((outputs.size() - 1) % 2 != 0) {
      SHERPA_ONNX_LOGE("ForwardLLM: invalid KV cache outputs size=%d",
                       static_cast<int>(outputs.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t inferred_layers = static_cast<int32_t>((outputs.size() - 1) / 2);
    if (inferred_layers != num_layers_) {
      SHERPA_ONNX_LOGE("ForwardLLM: KV outputs layers mismatch: expected=%d, got=%d",
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

  // Apply KV delta in-place to the KV cache.
  // Copy key_delta/value_delta into cache_key/value at positions [pos0:pos0+S)
  void ApplyKvDeltaInplace(std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
                          const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
                          const Ort::Value &cache_position) const {
    if (!cache_kv || cache_kv->size() != static_cast<size_t>(num_layers_) ||
        kv_delta.size() != static_cast<size_t>(num_layers_)) {
      SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: invalid kv sizes: cache=%zu delta=%zu",
                       cache_kv ? cache_kv->size() : 0, kv_delta.size());
      SHERPA_ONNX_EXIT(-1);
    }

    // cache_position: [S], first element is pos0 (contiguous write)
    auto pos_info = cache_position.GetTensorTypeAndShapeInfo();
    auto pos_shape = pos_info.GetShape();
    int64_t S = pos_shape.empty() ? 0 : pos_shape[0];
    if (S <= 0) {
      SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: cache_position has invalid shape");
      SHERPA_ONNX_EXIT(-1);
    }

    const int64_t *pos_data = cache_position.GetTensorData<int64_t>();
    int64_t pos0 = pos_data[0];

    if (pos0 < 0) {
      SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: pos0 < 0 (%lld)", static_cast<long long>(pos0));
      SHERPA_ONNX_EXIT(-1);
    }
    if (pos0 + S > max_total_len_) {
      SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: pos0+S exceeds max_total_len_ (%lld + %lld > %d), clamping S",
                       static_cast<long long>(pos0), static_cast<long long>(S), max_total_len_);
      S = max_total_len_ - pos0;
      if (S <= 0) return;
    }

    for (int32_t layer = 0; layer < num_layers_; ++layer) {
      Ort::Value &cache_key = (*cache_kv)[layer].first;
      Ort::Value &cache_val = (*cache_kv)[layer].second;

      const Ort::Value &delta_key = kv_delta[layer].first;
      const Ort::Value &delta_val = kv_delta[layer].second;

      auto ck_info = cache_key.GetTensorTypeAndShapeInfo();
      auto dk_info = delta_key.GetTensorTypeAndShapeInfo();

      auto ck_shape = ck_info.GetShape();  // [B, max_total_len, kv_h, hd]
      auto dk_shape = dk_info.GetShape();  // [B, S, kv_h, hd]

      int64_t B = ck_shape[0];
      int64_t kv_h = ck_shape[2];
      int64_t hd = ck_shape[3];

      // bytes per element
      auto elem_type = ck_info.GetElementType();
      size_t elem_bytes = 0;
      switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          elem_bytes = 4;
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
          elem_bytes = 2;
          break;
        default:
          SHERPA_ONNX_LOGE("ApplyKvDeltaInplace: unsupported elem_type=%d",
                           elem_type);
          SHERPA_ONNX_EXIT(-1);
      }

      size_t bytes_per_pos =
          static_cast<size_t>(kv_h) * static_cast<size_t>(hd) * elem_bytes;

      void *dst_k = cache_key.GetTensorMutableData<void>();
      void *dst_v = cache_val.GetTensorMutableData<void>();
      const void *src_k = delta_key.GetTensorData<void>();
      const void *src_v = delta_val.GetTensorData<void>();

      for (int64_t b = 0; b < B; ++b) {
        size_t dst_off =
            (static_cast<size_t>(b) * static_cast<size_t>(max_total_len_) +
             static_cast<size_t>(pos0)) *
            bytes_per_pos;
        size_t src_off =
            (static_cast<size_t>(b) * static_cast<size_t>(dk_shape[1])) *
            bytes_per_pos;

        size_t copy_bytes = static_cast<size_t>(S) * bytes_per_pos;

        uint8_t *dst_k_ptr = static_cast<uint8_t *>(dst_k) + dst_off;
        uint8_t *dst_v_ptr = static_cast<uint8_t *>(dst_v) + dst_off;
        const uint8_t *src_k_ptr = static_cast<const uint8_t *>(src_k) + src_off;
        const uint8_t *src_v_ptr = static_cast<const uint8_t *>(src_v) + src_off;

        std::memcpy(dst_k_ptr, src_k_ptr, copy_bytes);
        std::memcpy(dst_v_ptr, src_v_ptr, copy_bytes);
      }
    }
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
  int32_t GetMaxTotalLen() const { return max_total_len_; }
  int32_t LfrWindowSize() const { return lfr_window_size_; }
  int32_t LfrWindowShift() const { return lfr_window_shift_; }
  OrtAllocator *Allocator() { return allocator_; }
  bool HasEmbeddingModel() const { return has_embedding_model_; }
  bool UseKVCache() const { return true; }
  bool IsCpuProvider() const { return is_cpu_provider_; }

 private:
  void CheckFp16OnCuda() {
    if (use_cuda_iobinding_) {
      Ort::ModelMetadata meta_data = llm_sess_->GetModelMetadata();
      Ort::AllocatorWithDefaultOptions allocator;
      auto quant_type = LookupCustomModelMetaData(meta_data, "quantization_type", allocator);

      if (!quant_type.empty() && quant_type == "fp16") {
        SHERPA_ONNX_LOGE(
            "fp16 LLM models are not supported on CUDA yet. Please use "
            "fp32/int8 models.");
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

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

  void InitLLM(const std::string &model_path) {
    // For fp32 models: check for .data file by replacing .onnx with .data
    // int8 and fp16 models don't have .data files, so no need to check
    std::string data_path = model_path;
    if (data_path.size() >= 5 && data_path.substr(data_path.size() - 5) == ".onnx") {
      data_path = data_path.substr(0, data_path.size() - 5) + ".data";
    } else {
      data_path = model_path + ".data";
    }
    bool has_external_data = FileExists(data_path);

    // Resolve absolute path for model file
    std::string abs_model_path = model_path;
    if (!model_path.empty() && model_path[0] != '/') {
      char abs_path_buf[PATH_MAX];
      if (realpath(model_path.c_str(), abs_path_buf) != nullptr) {
        abs_model_path = abs_path_buf;
      }
    }

    if (has_external_data) {
      // When external data exists, use absolute file path to create session.
      // ONNX Runtime will automatically find .data file in the same directory
      // as the model file when using absolute path.
      llm_sess_ =
          std::make_unique<Ort::Session>(env_, SHERPA_ONNX_TO_ORT_PATH(abs_model_path), sess_opts_llm_);
    } else {
      // No external data: load entire model into memory
      std::vector<char> model_data = ReadFile(model_path);
      llm_sess_ = std::make_unique<Ort::Session>(env_, model_data.data(),
                                                 model_data.size(),
                                                 sess_opts_llm_);
    }

    GetInputNames(llm_sess_.get(), &llm_input_names_, &llm_input_names_ptr_);
    GetOutputNames(llm_sess_.get(), &llm_output_names_, &llm_output_names_ptr_);

    llm_embeds_in_type_ = GetSessionInputElemType(llm_sess_.get(), 0);
    if (llm_embeds_in_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      SHERPA_ONNX_LOGE("LLM inputs_embeds must be float32, got elem_type=%d",
                       (int)llm_embeds_in_type_);
      SHERPA_ONNX_EXIT(-1);
    }

    Ort::ModelMetadata meta_data = llm_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("LLM model metadata:\n%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("LLM model metadata:\n%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }

    int32_t num_outputs = static_cast<int32_t>(llm_output_names_.size());
    if (num_outputs < 1 || (num_outputs - 1) % 2 != 0) {
      SHERPA_ONNX_LOGE(
          "LLM model must have 1 logits output + 2*num_layers KV outputs, got %d outputs",
          num_outputs);
      SHERPA_ONNX_EXIT(-1);
    }
    int32_t inferred_layers = (num_outputs - 1) / 2;

    auto num_layers_value =
        LookupCustomModelMetaData(meta_data, "num_layers", allocator);
    if (!num_layers_value.empty()) {
      num_layers_ = atoi(num_layers_value.c_str());
      if (num_layers_ <= 0) {
        SHERPA_ONNX_LOGE("Invalid num_layers=%d from metadata", num_layers_);
        SHERPA_ONNX_EXIT(-1);
      }
      if (num_layers_ != inferred_layers) {
        SHERPA_ONNX_LOGE("LLM num_layers mismatch: metadata=%d, inferred=%d",
                         num_layers_, inferred_layers);
        SHERPA_ONNX_EXIT(-1);
      }
    } else {
      num_layers_ = inferred_layers;
    }

    // Read max_total_len from metadata
    SHERPA_ONNX_READ_META_DATA(max_total_len_, "max_total_len");
    if (max_total_len_ <= 0) {
      SHERPA_ONNX_LOGE("Invalid max_total_len=%d from metadata", max_total_len_);
      SHERPA_ONNX_EXIT(-1);
    }

    // Detect KV delta model type (model_type metadata should contain "kv_delta")
    auto model_type_value =
        LookupCustomModelMetaData(meta_data, "model_type", allocator);
    is_kv_delta_model_ = (!model_type_value.empty() &&
                          model_type_value.find("kv_delta") != std::string::npos);

    if (!is_kv_delta_model_) {
      SHERPA_ONNX_LOGE("Only KV delta models are supported, but model_type does not contain 'kv_delta'");
      SHERPA_ONNX_EXIT(-1);
    }

    // Validate input layout: 0 embeds, 1 attention_mask, 2 cache_position, 3+ KV cache
    if (llm_input_names_.size() < 3u) {
      SHERPA_ONNX_LOGE("LLM model inputs must be >=3 (embeds,mask,cache_position)");
      SHERPA_ONNX_EXIT(-1);
    }

    cache_position_input_index_ = 2;
    past_kv_input_start_index_ = 3;

    int32_t expected_inputs = 3 + 2 * num_layers_;
    int32_t actual_inputs = static_cast<int32_t>(llm_input_names_.size());
    if (actual_inputs != expected_inputs) {
      if (actual_inputs == 2 + 2 * num_layers_) {
        SHERPA_ONNX_LOGE(
            "LLM model inputs mismatch: expected %d (=3+2*num_layers with cache_position) "
            "got %d (=2+2*num_layers without cache_position). "
            "Please use a model exported with cache_position support.",
            expected_inputs, actual_inputs);
      } else {
        SHERPA_ONNX_LOGE(
            "LLM model inputs mismatch: expected %d (=3+2*num_layers) got %d",
            expected_inputs, actual_inputs);
      }
      SHERPA_ONNX_EXIT(-1);
    }

    kv_in_type_ = GetSessionInputElemType(llm_sess_.get(), past_kv_input_start_index_);
    kv_in_type_v_ = GetSessionInputElemType(llm_sess_.get(), past_kv_input_start_index_ + 1);
    if (!(kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          kv_in_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
      SHERPA_ONNX_LOGE("LLM past_key elem_type=%d not supported", (int)kv_in_type_);
      SHERPA_ONNX_EXIT(-1);
    }
    if (!(kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
          kv_in_type_v_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)) {
      SHERPA_ONNX_LOGE("LLM past_value elem_type=%d not supported", (int)kv_in_type_v_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto past_key_ti = llm_sess_->GetInputTypeInfo(past_kv_input_start_index_);
    past_key_shape_tpl_ = past_key_ti.GetTensorTypeAndShapeInfo().GetShape();

    auto past_value_ti = llm_sess_->GetInputTypeInfo(past_kv_input_start_index_ + 1);
    past_value_shape_tpl_ = past_value_ti.GetTensorTypeAndShapeInfo().GetShape();
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
  std::unique_ptr<Ort::Session> llm_sess_;
  std::unique_ptr<Ort::Session> embedding_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;
  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> llm_input_names_;
  std::vector<const char *> llm_input_names_ptr_;
  std::vector<std::string> llm_output_names_;
  std::vector<const char *> llm_output_names_ptr_;

  std::vector<std::string> embedding_input_names_;
  std::vector<const char *> embedding_input_names_ptr_;
  std::vector<std::string> embedding_output_names_;
  std::vector<const char *> embedding_output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t hidden_size_ = 0;
  int32_t lfr_window_size_ = 0;
  int32_t lfr_window_shift_ = 0;

  int32_t num_layers_ = 0;
  int32_t max_total_len_ = 0;  // attention_mask length / cache capacity
  bool has_embedding_model_ = false;

  ONNXTensorElementDataType encoder_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType llm_embeds_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  // KV input element types (for CreateEmptyKVCache).
  ONNXTensorElementDataType kv_in_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  ONNXTensorElementDataType kv_in_type_v_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

  // Input indices for KV cache LLM.
  size_t cache_position_input_index_ = 2;
  size_t past_kv_input_start_index_ = 3;

  std::vector<int64_t> past_key_shape_tpl_;
  std::vector<int64_t> past_value_shape_tpl_;

  bool is_cpu_provider_ = false;
  bool is_kv_delta_model_ = false;
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
OfflineFunASRNanoModel::ForwardLLM(
    Ort::Value inputs_embeds, Ort::Value attention_mask,
    const Ort::Value &cache_position,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &cache_kv) {
  return impl_->ForwardLLM(std::move(inputs_embeds), std::move(attention_mask),
                           std::move(cache_position), cache_kv);
}

std::vector<std::pair<Ort::Value, Ort::Value>>
OfflineFunASRNanoModel::CreateEmptyKVCache(int64_t batch) {
  return impl_->CreateEmptyKVCache(batch);
}

void OfflineFunASRNanoModel::ApplyKvDeltaInplace(
    std::vector<std::pair<Ort::Value, Ort::Value>> *cache_kv,
    const std::vector<std::pair<Ort::Value, Ort::Value>> &kv_delta,
    const Ort::Value &cache_position) {
  return impl_->ApplyKvDeltaInplace(cache_kv, kv_delta, cache_position);
}

bool OfflineFunASRNanoModel::UseKVCache() const { return impl_->UseKVCache(); }

Ort::Value OfflineFunASRNanoModel::ForwardEmbedding(Ort::Value input_ids) {
  return impl_->ForwardEmbedding(std::move(input_ids));
}

int32_t OfflineFunASRNanoModel::VocabSize() const { return impl_->VocabSize(); }
int32_t OfflineFunASRNanoModel::HiddenSize() const { return impl_->HiddenSize(); }
int32_t OfflineFunASRNanoModel::GetMaxTotalLen() const { return impl_->GetMaxTotalLen(); }

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

#if __ANDROID_API__ >= 9
template OfflineFunASRNanoModel::OfflineFunASRNanoModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineFunASRNanoModel::OfflineFunASRNanoModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
