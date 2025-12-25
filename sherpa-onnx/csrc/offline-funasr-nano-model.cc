// sherpa-onnx/csrc/offline-funasr-nano-model.cc

#include "sherpa-onnx/csrc/offline-funasr-nano-model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
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

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

namespace {

// Convert IEEE 754 half-precision (16-bit) float to single-precision (32-bit)
// float. Handles special cases: zero, subnormal, normal, infinity, and NaN.
static inline float HalfBitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  const uint32_t exp  = (h & 0x7C00u) >> 10;
  const uint32_t mant = (h & 0x03FFu);
  uint32_t fbits = 0;
  if (exp == 0) {
    if (mant == 0) {
      fbits = sign;
    } else {
      uint32_t m = mant;
      uint32_t e = 127 - 15 + 1;
      while ((m & 0x0400u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03FFu;
      fbits = sign | (e << 23) | (m << 13);
    }
  } else if (exp == 31) {
    fbits = sign | 0x7F800000u | (mant << 13);
  } else {
    const uint32_t e = exp + (127 - 15);
    fbits = sign | (e << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &fbits, sizeof(out));
  return out;
}

// Convert IEEE 754 single-precision (32-bit) float to half-precision (16-bit)
// float. Handles overflow (clamped to infinity), underflow (clamped to zero),
// and normal values with proper rounding.
static inline uint16_t FloatToHalfBits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFFu);
  const uint32_t mant = x & 0x007FFFFFu;
  if (exp == 255) {
    if (mant == 0) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x1u : 0));
  }
  int32_t new_exp = exp - 127 + 15;
  if (new_exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  } else if (new_exp <= 0) {
    if (new_exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    uint32_t m = mant | 0x00800000u;
    int32_t shift = 14 - new_exp;
    uint32_t half_m = m >> shift;
    if ((m >> (shift - 1)) & 1u) {
      half_m += 1;
    }
    return static_cast<uint16_t>(sign | (half_m & 0x03FFu));
  } else {
    uint16_t half_exp = static_cast<uint16_t>(new_exp << 10);
    uint32_t half_m   = mant >> 13;
    if (mant & 0x00001000u) {
      half_m += 1;
      if (half_m == 0x0400u) {
        half_m = 0;
        half_exp = static_cast<uint16_t>((new_exp + 1) << 10);
        if ((half_exp >> 10) >= 31) {
          return static_cast<uint16_t>(sign | 0x7C00u);
        }
      }
    }
    return static_cast<uint16_t>(sign | half_exp | (half_m & 0x03FFu));
  }
}

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

// Helper template to convert std::string or const char* to const char*.
template <typename T>
static inline const char *ToCStr(const T &x) {
  return x.c_str();
}

template <>
inline const char *ToCStr<const char *>(const char *const &x) {
  return x;
}

// Check if a tensor is stored on CUDA device.
static inline bool TensorIsCuda(const Ort::Value &v) {
  if (!v.IsTensor()) return false;
  auto mi = v.GetTensorMemoryInfo();
  auto name_obj = mi.GetAllocatorName();
  const char *name = ToCStr(name_obj);
  if (!name || !name[0]) return false;
  return std::strstr(name, "Cuda") || std::strstr(name, "CUDA");
}

// Copy tensor data from GPU to CPU if needed.
// Uses dynamic loading of CUDA functions to avoid compile-time dependencies.
static inline void CopyRawToCpu(void *dst, const Ort::Value &src, size_t bytes,
                                bool force_cpu = false) {
  const void *p = src.GetTensorData<void>();
  if (bytes == 0 || dst == p) return;

  if (force_cpu || !TensorIsCuda(src)) {
    std::memcpy(dst, p, bytes);
    return;
  }

  typedef int (*cudaMemcpyFunc)(void *, const void *, size_t, int);
  typedef int (*cudaDeviceSynchronizeFunc)();
  typedef int (*cudaGetLastErrorFunc)();

  static void *cuda_handle = nullptr;
  static cudaMemcpyFunc cudaMemcpy_dyn = nullptr;
  static cudaDeviceSynchronizeFunc cudaDeviceSynchronize_dyn = nullptr;
  static cudaGetLastErrorFunc cudaGetLastError_dyn = nullptr;
  static bool cuda_checked = false;

  if (!cuda_checked) {
    cuda_checked = true;
    cuda_handle = dlopen("libcudart.so", RTLD_LAZY);
    if (!cuda_handle) cuda_handle = dlopen("libcudart.so.12", RTLD_LAZY);
    if (!cuda_handle) cuda_handle = dlopen("libcudart.so.11", RTLD_LAZY);
    if (cuda_handle) {
      cudaMemcpy_dyn = (cudaMemcpyFunc)dlsym(cuda_handle, "cudaMemcpy");
      cudaDeviceSynchronize_dyn =
          (cudaDeviceSynchronizeFunc)dlsym(cuda_handle, "cudaDeviceSynchronize");
      cudaGetLastError_dyn =
          (cudaGetLastErrorFunc)dlsym(cuda_handle, "cudaGetLastError");
      if (!cudaMemcpy_dyn) {
        dlclose(cuda_handle);
        cuda_handle = nullptr;
        cudaMemcpy_dyn = nullptr;
        cudaDeviceSynchronize_dyn = nullptr;
        cudaGetLastError_dyn = nullptr;
      }
    }
  }

  if (!cudaMemcpy_dyn) {
    SHERPA_ONNX_LOGE("Tensor is on CUDA but libcudart/cudaMemcpy is unavailable");
    SHERPA_ONNX_EXIT(-1);
  }

  if (cudaGetLastError_dyn) cudaGetLastError_dyn();
  int err = cudaMemcpy_dyn(dst, p, bytes, 2);
  if (err == 0) {
    if (cudaDeviceSynchronize_dyn) cudaDeviceSynchronize_dyn();
    return;
  }

  if (cudaGetLastError_dyn) cudaGetLastError_dyn();
  SHERPA_ONNX_LOGE("cudaMemcpy(DeviceToHost) failed");
  SHERPA_ONNX_EXIT(-1);
}

// Get the element type of a session input tensor.
static inline ONNXTensorElementDataType GetSessionInputElemType(Ort::Session *sess, size_t input_index) {
  auto ti = sess->GetInputTypeInfo(input_index);
  auto t  = ti.GetTensorTypeAndShapeInfo();
  return static_cast<ONNXTensorElementDataType>(t.GetElementType());
}

// Get the element type of a session output tensor.
static inline ONNXTensorElementDataType GetSessionOutputElemType(Ort::Session *sess, size_t output_index) {
  auto ti = sess->GetOutputTypeInfo(output_index);
  auto t  = ti.GetTensorTypeAndShapeInfo();
  return static_cast<ONNXTensorElementDataType>(t.GetElementType());
}

template <typename T>
static Ort::Value AllocTensor(OrtAllocator *alloc, const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor<T>(alloc, shape.data(), shape.size());
}

template <>
Ort::Value AllocTensor<uint16_t>(OrtAllocator *alloc,
                                  const std::vector<int64_t> &shape) {
  return Ort::Value::CreateTensor(alloc, shape.data(), shape.size(),
                                  ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
}

// Convert tensor to float32, handling both float16 and float32 inputs.
// If force_cpu is true, ensures the output tensor is allocated on CPU.
static Ort::Value CastToFloat32(Ort::Value in, OrtAllocator *alloc,
                                bool force_cpu = false) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  size_t n = NumelFromShape(shape);
  if (n == 0) return in;
  auto et = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  Ort::Value out = AllocTensor<float>(alloc, shape);
  float *dst = out.GetTensorMutableData<float>();
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    CopyRawToCpu(dst, in, n * sizeof(float), force_cpu);
    return out;
  }
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
      et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    std::vector<uint16_t> tmp(n);
    CopyRawToCpu(tmp.data(), in, n * sizeof(uint16_t), force_cpu);
    for (size_t i = 0; i < n; ++i) dst[i] = HalfBitsToFloat(tmp[i]);
    return out;
  }
  SHERPA_ONNX_LOGE("CastToFloat32: unsupported input elem_type=%d", (int)et);
  return in;
}

// Convert tensor to float16, handling both float16 and float32 inputs.
static Ort::Value CastToFloat16(Ort::Value in, OrtAllocator *alloc) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  size_t n = NumelFromShape(shape);
  if (n == 0) return in;
  auto et = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  Ort::Value out = AllocTensor<uint16_t>(alloc, shape);
  uint16_t *dst = out.GetTensorMutableData<uint16_t>();
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
      et == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
    CopyRawToCpu(dst, in, n * sizeof(uint16_t));
    return out;
  }
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> tmp(n);
    CopyRawToCpu(tmp.data(), in, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) dst[i] = FloatToHalfBits(tmp[i]);
    return out;
  }
  SHERPA_ONNX_LOGE("CastToFloat16: unsupported input elem_type=%d", (int)et);
  return in;
}

// Cast tensor to the expected element type (float16 or float32).
// Returns the input unchanged if it already matches the expected type.
static Ort::Value CastFloatLikeForExpected(Ort::Value in,
                                           ONNXTensorElementDataType expected,
                                           OrtAllocator *alloc,
                                           bool force_cpu = false) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto actual = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (actual == expected) return in;
  if (expected == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return CastToFloat16(std::move(in), alloc);
  }
  if (expected == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return CastToFloat32(std::move(in), alloc, force_cpu);
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
static Ort::Value CastMaskToInt64IfNeeded(Ort::Value in, OrtAllocator *alloc) {
  if (!in.IsTensor()) return in;
  auto info = in.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  size_t n = NumelFromShape(shape);
  if (n == 0) return in;
  auto et = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return in;

  if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> tmp(n);
    CopyRawToCpu(tmp.data(), in, n * sizeof(int32_t), false);
    Ort::Value out = AllocTensor<int64_t>(alloc, shape);
    int64_t *dst = out.GetTensorMutableData<int64_t>();
    for (size_t i = 0; i < n; ++i) dst[i] = static_cast<int64_t>(tmp[i]);
    return out;
  }

  SHERPA_ONNX_LOGE("attention_mask elem_type=%d not supported, expected int64",
                   (int)et);
  return in;
}

// Calculate the byte size of a tensor based on its shape and element type.
// Supports float16 and float32 types. Exits on unsupported types.
static size_t CalculateTensorBytes(const Ort::Value &tensor) {
  if (!tensor.IsTensor()) {
    SHERPA_ONNX_LOGE("CalculateTensorBytes: input is not a tensor");
    SHERPA_ONNX_EXIT(-1);
  }
  auto info = tensor.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  auto type = static_cast<ONNXTensorElementDataType>(info.GetElementType());
  size_t n = NumelFromShape(shape);
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    return n * sizeof(uint16_t);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return n * sizeof(float);
  }
  SHERPA_ONNX_LOGE("CalculateTensorBytes: unsupported tensor type %d", (int)type);
  SHERPA_ONNX_EXIT(-1);
  return 0; 
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
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.funasr_nano;
    InitEncoderAdaptor(c.encoder_adaptor);
    InitLLMPrefill(c.llm_prefill);
    InitLLMDecode(c.llm_decode);
    use_kv_cache_ = true;
    if (!c.embedding.empty()) {
      InitEmbedding(c.embedding);
      has_embedding_model_ = true;
    } else {
      has_embedding_model_ = false;
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
    encoder_out_type_ = GetSessionOutputElemType(encoder_sess_.get(), 0);
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
    prefill_out_type_ = GetSessionOutputElemType(prefill_sess_.get(), 0);
    Ort::ModelMetadata meta_data = prefill_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%s\n", os.str().c_str());
#endif
    }
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }
    SHERPA_ONNX_READ_META_DATA(num_layers_, "num_layers");
  }

  void InitLLMDecodeFromMemory(void *model_data, size_t model_data_length) {
    decode_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_llm_);
    GetInputNames(decode_sess_.get(), &decode_input_names_,
                  &decode_input_names_ptr_);
    GetOutputNames(decode_sess_.get(), &decode_output_names_,
                   &decode_output_names_ptr_);
    decode_embeds_in_type_ = GetSessionInputElemType(decode_sess_.get(), 0);
    decode_out_type_ = GetSessionOutputElemType(decode_sess_.get(), 0);
    Ort::ModelMetadata meta_data = decode_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Decode model metadata:\n%{public}s\n", os.str().c_str());
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
    int32_t decode_num_layers = 0;
    SHERPA_ONNX_READ_META_DATA(decode_num_layers, "num_layers");
    if (num_layers_ > 0 && decode_num_layers != num_layers_) {
      SHERPA_ONNX_LOGE(
          "Decode model num_layers (%d) != prefill num_layers (%d)",
          decode_num_layers, num_layers_);
    }
    if (num_layers_ == 0) num_layers_ = decode_num_layers;
  }

  void InitEmbeddingFromMemory(void *model_data, size_t model_data_length) {
    embedding_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_embedding_);
    GetInputNames(embedding_sess_.get(), &embedding_input_names_,
                  &embedding_input_names_ptr_);
    GetOutputNames(embedding_sess_.get(), &embedding_output_names_,
                   &embedding_output_names_ptr_);
    embedding_out_type_ =
        GetSessionOutputElemType(embedding_sess_.get(), 0);
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
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    const auto &c = config_.funasr_nano;
    auto buf_encoder = ReadFile(mgr, c.encoder_adaptor);
    InitEncoderAdaptorFromMemory(buf_encoder.data(), buf_encoder.size());
    auto buf_prefill = ReadFile(mgr, c.llm_prefill);
    InitLLMPrefillFromMemory(buf_prefill.data(), buf_prefill.size());
    auto buf_decode = ReadFile(mgr, c.llm_decode);
    InitLLMDecodeFromMemory(buf_decode.data(), buf_decode.size());
    use_kv_cache_ = true;
    if (!c.embedding.empty()) {
      auto buf_embedding = ReadFile(mgr, c.embedding);
      InitEmbeddingFromMemory(buf_embedding.data(), buf_embedding.size());
      has_embedding_model_ = true;
    } else {
      has_embedding_model_ = false;
    }
  }

  // Forward pass through encoder adaptor model.
  // Converts audio features to embeddings compatible with the LLM.
  Ort::Value ForwardEncoderAdaptor(Ort::Value features) {
    if (NeedsTypeConversion(features, encoder_in_type_)) {
      features = CastFloatLikeForExpected(std::move(features), encoder_in_type_,
                                         allocator_, is_cpu_provider_);
    }
    std::array<Ort::Value, 1> inputs = {std::move(features)};
    auto outputs = encoder_sess_->Run(
        {}, encoder_input_names_ptr_.data(), inputs.data(), inputs.size(),
        encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto out_et =
        static_cast<ONNXTensorElementDataType>(out_info.GetElementType());
    if (out_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      return CastToFloat16(std::move(outputs[0]), allocator_);
    }
    if (out_et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return CastToFloat32(std::move(outputs[0]), allocator_,
                           is_cpu_provider_);
    }
    return CastToFloat32(std::move(outputs[0]), allocator_, is_cpu_provider_);
  }

  Ort::Value ForwardLLM(Ort::Value, Ort::Value) {
    SHERPA_ONNX_LOGE(
        "ForwardLLM is not supported for FunASR-nano. Use KV cache "
        "prefill/decode.");
    SHERPA_ONNX_EXIT(-1);
  }

  // Forward pass through LLM prefill model with full context.
  // Returns logits and initial KV cache states for all layers.
  std::pair<Ort::Value, std::vector<std::pair<Ort::Value, Ort::Value>>>
  ForwardLLMPrefill(Ort::Value inputs_embeds, Ort::Value attention_mask) {
    if (!use_kv_cache_) {
      SHERPA_ONNX_LOGE(
          "ForwardLLMPrefill called but KV cache mode is not enabled.");
      SHERPA_ONNX_EXIT(-1);
    }
    if (NeedsTypeConversion(inputs_embeds, prefill_embeds_in_type_)) {
      inputs_embeds = CastFloatLikeForExpected(
          std::move(inputs_embeds), prefill_embeds_in_type_, allocator_,
          is_cpu_provider_);
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
    std::array<Ort::Value, 2> inputs = {std::move(inputs_embeds),
                                       std::move(attention_mask)};
    auto outputs = prefill_sess_->Run(
        {}, prefill_input_names_ptr_.data(), inputs.data(), inputs.size(),
        prefill_output_names_ptr_.data(), prefill_output_names_ptr_.size());
    // First output is logits, remaining outputs are past_key_values
    Ort::Value logits = std::move(outputs[0]);
    logits = CastToFloat32(std::move(logits), allocator_, is_cpu_provider_);
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
  ForwardLLMDecode(Ort::Value inputs_embeds, Ort::Value attention_mask,
                   const std::vector<std::pair<Ort::Value, Ort::Value>>
                       &past_key_values) {
    if (!use_kv_cache_) {
      SHERPA_ONNX_LOGE(
          "ForwardLLMDecode called but KV cache mode is not enabled.");
      SHERPA_ONNX_EXIT(-1);
    }
    if (NeedsTypeConversion(inputs_embeds, decode_embeds_in_type_)) {
      inputs_embeds = CastFloatLikeForExpected(
          std::move(inputs_embeds), decode_embeds_in_type_, allocator_,
          is_cpu_provider_);
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
    // Build inputs: [inputs_embeds, attention_mask, past_key_0, past_value_0, ...]
    // NOTE: We create non-owning Ort::Value views that reference existing buffers.
    std::vector<Ort::Value> inputs;
    inputs.reserve(2 + 2 * past_key_values.size());
    inputs.push_back(std::move(inputs_embeds));
    inputs.push_back(std::move(attention_mask));
    for (const auto &kv : past_key_values) {
      auto k_info = kv.first.GetTensorTypeAndShapeInfo();
      auto k_shape = k_info.GetShape();
      auto k_type =
          static_cast<ONNXTensorElementDataType>(k_info.GetElementType());
      size_t k_bytes = CalculateTensorBytes(kv.first);
      auto v_info = kv.second.GetTensorTypeAndShapeInfo();
      auto v_shape = v_info.GetShape();
      auto v_type =
          static_cast<ONNXTensorElementDataType>(v_info.GetElementType());
      size_t v_bytes = CalculateTensorBytes(kv.second);
      auto k_mem = kv.first.GetTensorMemoryInfo();
      auto v_mem = kv.second.GetTensorMemoryInfo();
      inputs.push_back(Ort::Value::CreateTensor(
          k_mem, const_cast<void *>(kv.first.GetTensorData<void>()), k_bytes,
          k_shape.data(), k_shape.size(), k_type));
      inputs.push_back(Ort::Value::CreateTensor(
          v_mem, const_cast<void *>(kv.second.GetTensorData<void>()), v_bytes,
          v_shape.data(), v_shape.size(), v_type));
    }
    // Build input names: [inputs_embeds, attention_mask, past_key_0, past_value_0, ...]
    std::vector<const char *> input_names_ptr;
    input_names_ptr.reserve(2 + 2 * past_key_values.size());
    input_names_ptr.push_back(decode_input_names_ptr_[0]);
    input_names_ptr.push_back(decode_input_names_ptr_[1]);
    for (size_t i = 0; i < past_key_values.size(); ++i) {
      input_names_ptr.push_back(decode_input_names_ptr_[2 + 2 * i]);
      input_names_ptr.push_back(decode_input_names_ptr_[2 + 2 * i + 1]);
    }
    auto outputs = decode_sess_->Run(
        {}, input_names_ptr.data(), inputs.data(), inputs.size(),
        decode_output_names_ptr_.data(), decode_output_names_ptr_.size());
    // First output is logits, remaining outputs are updated past_key_values
    Ort::Value logits = std::move(outputs[0]);
    logits = CastToFloat32(std::move(logits), allocator_, is_cpu_provider_);
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
    if (!has_embedding_model_) {
      SHERPA_ONNX_LOGE("Embedding model is not loaded");
      SHERPA_ONNX_EXIT(-1);
    }
    std::array<Ort::Value, 1> inputs = {std::move(input_ids)};
    auto outputs = embedding_sess_->Run(
        {}, embedding_input_names_ptr_.data(), inputs.data(), inputs.size(),
        embedding_output_names_ptr_.data(), embedding_output_names_ptr_.size());
    if (NeedsTypeConversion(outputs[0], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
      return CastToFloat32(std::move(outputs[0]), allocator_,
                           is_cpu_provider_);
    }
    return std::move(outputs[0]);
  }

  int32_t VocabSize() const { return vocab_size_; }
  int32_t HiddenSize() const { return hidden_size_; }
  int32_t LfrWindowSize() const { return lfr_window_size_; }
  int32_t LfrWindowShift() const { return lfr_window_shift_; }
  OrtAllocator *Allocator() { return allocator_; }
  bool HasEmbeddingModel() const { return has_embedding_model_; }
  bool UseKVCache() const { return use_kv_cache_; }
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
        env_, model_path.c_str(), sess_opts_encoder_);
    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);
    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);
    encoder_in_type_ = GetSessionInputElemType(encoder_sess_.get(), 0);
    encoder_out_type_ = GetSessionOutputElemType(encoder_sess_.get(), 0);
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
        env_, model_path.c_str(), sess_opts_llm_);
    GetInputNames(prefill_sess_.get(), &prefill_input_names_,
                  &prefill_input_names_ptr_);
    GetOutputNames(prefill_sess_.get(), &prefill_output_names_,
                   &prefill_output_names_ptr_);
    prefill_embeds_in_type_ = GetSessionInputElemType(prefill_sess_.get(), 0);
    prefill_out_type_ = GetSessionOutputElemType(prefill_sess_.get(), 0);
    Ort::ModelMetadata meta_data = prefill_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("Prefill model metadata:\n%s\n", os.str().c_str());
#endif
    }
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    if (hidden_size_ == 0) {
      SHERPA_ONNX_READ_META_DATA(hidden_size_, "hidden_size");
    }
    SHERPA_ONNX_READ_META_DATA(num_layers_, "num_layers");
  }

  void InitLLMDecode(const std::string &model_path) {
    decode_sess_ = std::make_unique<Ort::Session>(
        env_, model_path.c_str(), sess_opts_llm_);
    GetInputNames(decode_sess_.get(), &decode_input_names_,
                  &decode_input_names_ptr_);
    GetOutputNames(decode_sess_.get(), &decode_output_names_,
                   &decode_output_names_ptr_);
    decode_embeds_in_type_ = GetSessionInputElemType(decode_sess_.get(), 0);
    decode_out_type_ = GetSessionOutputElemType(decode_sess_.get(), 0);
    Ort::ModelMetadata meta_data = decode_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("Decode model metadata:\n%{public}s\n", os.str().c_str());
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
    int32_t decode_num_layers = 0;
    SHERPA_ONNX_READ_META_DATA(decode_num_layers, "num_layers");
    if (num_layers_ > 0 && decode_num_layers != num_layers_) {
      SHERPA_ONNX_LOGE(
          "Decode model num_layers (%d) != prefill num_layers (%d)",
          decode_num_layers, num_layers_);
    }
    if (num_layers_ == 0) num_layers_ = decode_num_layers;
  }

  void InitEmbedding(const std::string &model_path) {
    embedding_sess_ = std::make_unique<Ort::Session>(
        env_, model_path.c_str(), sess_opts_embedding_);
    GetInputNames(embedding_sess_.get(), &embedding_input_names_,
                  &embedding_input_names_ptr_);
    GetOutputNames(embedding_sess_.get(), &embedding_output_names_,
                   &embedding_output_names_ptr_);
    embedding_out_type_ =
        GetSessionOutputElemType(embedding_sess_.get(), 0);
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
  int32_t num_layers_ = 0;
  int32_t lfr_window_size_ = 0;
  int32_t lfr_window_shift_ = 0;
  bool has_embedding_model_ = false;
  bool use_kv_cache_ = true;

  ONNXTensorElementDataType encoder_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType encoder_out_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType prefill_embeds_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType prefill_out_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType decode_embeds_in_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType decode_out_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType embedding_out_type_ =
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

Ort::Value OfflineFunASRNanoModel::ForwardLLM(Ort::Value inputs_embeds,
                                              Ort::Value attention_mask) {
  return impl_->ForwardLLM(std::move(inputs_embeds),
                           std::move(attention_mask));
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

bool OfflineFunASRNanoModel::UseKVCache() const {
  return impl_->UseKVCache();
}

Ort::Value OfflineFunASRNanoModel::ForwardEmbedding(Ort::Value input_ids) {
  return impl_->ForwardEmbedding(std::move(input_ids));
}

int32_t OfflineFunASRNanoModel::VocabSize() const {
  return impl_->VocabSize();
}
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
