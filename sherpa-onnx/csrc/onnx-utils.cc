// sherpa-onnx/csrc/onnx-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo
#include "sherpa-onnx/csrc/onnx-utils.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

static std::string GetInputName(Ort::Session *sess, size_t index,
                                OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetInputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetInputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

static std::string GetOutputName(Ort::Session *sess, size_t index,
                                 OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = sess->GetOutputNameAllocated(index, allocator);
  return v.get();
#else
  auto v = sess->GetOutputName(index, allocator);
  std::string ans = v;
  allocator->Free(allocator, v);
  return ans;
#endif
}

void GetInputNames(Ort::Session *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetInputCount();
  input_names->resize(node_count);
  input_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*input_names)[i] = GetInputName(sess, i, allocator);
    (*input_names_ptr)[i] = (*input_names)[i].c_str();
  }
}

void GetOutputNames(Ort::Session *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetOutputCount();
  output_names->resize(node_count);
  output_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    (*output_names)[i] = GetOutputName(sess, i, allocator);
    (*output_names_ptr)[i] = (*output_names)[i].c_str();
  }
}

Ort::Value GetEncoderOutFrame(OrtAllocator *allocator, Ort::Value *encoder_out,
                              int32_t t) {
  std::vector<int64_t> encoder_out_shape =
      encoder_out->GetTensorTypeAndShapeInfo().GetShape();

  auto batch_size = encoder_out_shape[0];
  auto num_frames = encoder_out_shape[1];
  assert(t < num_frames);

  auto encoder_out_dim = encoder_out_shape[2];

  auto offset = num_frames * encoder_out_dim;

  std::array<int64_t, 2> shape{batch_size, encoder_out_dim};

  Ort::Value ans =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  float *dst = ans.GetTensorMutableData<float>();
  const float *src = encoder_out->GetTensorData<float>();

  for (int32_t i = 0; i != batch_size; ++i) {
    std::copy(src + t * encoder_out_dim, src + (t + 1) * encoder_out_dim, dst);
    src += offset;
    dst += encoder_out_dim;
  }
  return ans;
}

void PrintModelMetadata(std::ostream &os, const Ort::ModelMetadata &meta_data) {
  Ort::AllocatorWithDefaultOptions allocator;
#if ORT_API_VERSION >= 12
  std::vector<Ort::AllocatedStringPtr> v =
      meta_data.GetCustomMetadataMapKeysAllocated(allocator);
  for (const auto &key : v) {
    auto p = meta_data.LookupCustomMetadataMapAllocated(key.get(), allocator);
    os << key.get() << "=" << p.get() << "\n";
  }
#else
  int64_t num_keys = 0;
  char **keys = meta_data.GetCustomMetadataMapKeys(allocator, num_keys);
  for (int32_t i = 0; i < num_keys; ++i) {
    auto v = LookupCustomModelMetaData(meta_data, keys[i], allocator);
    os << keys[i] << "=" << v << "\n";
    allocator.Free(keys[i]);
  }

  allocator.Free(keys);
#endif
}

Ort::Value Clone(OrtAllocator *allocator, const Ort::Value *v) {
  auto type_and_shape = v->GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_and_shape.GetShape();

  switch (type_and_shape.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      Ort::Value ans = Ort::Value::CreateTensor<int32_t>(
          allocator, shape.data(), shape.size());
      const int32_t *start = v->GetTensorData<int32_t>();
      const int32_t *end = start + type_and_shape.GetElementCount();
      int32_t *dst = ans.GetTensorMutableData<int32_t>();
      std::copy(start, end, dst);
      return ans;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      Ort::Value ans = Ort::Value::CreateTensor<int64_t>(
          allocator, shape.data(), shape.size());
      const int64_t *start = v->GetTensorData<int64_t>();
      const int64_t *end = start + type_and_shape.GetElementCount();
      int64_t *dst = ans.GetTensorMutableData<int64_t>();
      std::copy(start, end, dst);
      return ans;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, shape.data(),
                                                       shape.size());
      const float *start = v->GetTensorData<float>();
      const float *end = start + type_and_shape.GetElementCount();
      float *dst = ans.GetTensorMutableData<float>();
      std::copy(start, end, dst);
      return ans;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      Ort::Value ans =
          Ort::Value::CreateTensor(allocator, shape.data(), shape.size(),
                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
      const auto *start = v->GetTensorData<uint16_t>();
      const auto *end = start + type_and_shape.GetElementCount();
      auto *dst = ans.GetTensorMutableData<uint16_t>();
      std::copy(start, end, dst);
      return ans;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      Ort::Value ans = Ort::Value::CreateTensor<uint16_t>(
          allocator, shape.data(), shape.size());
      const auto *start = v->GetTensorData<uint16_t>();
      const auto *end = start + type_and_shape.GetElementCount();
      auto *dst = ans.GetTensorMutableData<uint16_t>();
      std::copy(start, end, dst);
      return ans;
    }

    default:
      SHERPA_ONNX_LOGE("Unsupported type: %d\n",
                       static_cast<int32_t>(type_and_shape.GetElementType()));
      SHERPA_ONNX_EXIT(-1);
      // unreachable code
      return Ort::Value{nullptr};
  }
}

Ort::Value View(Ort::Value *v) {
  auto type_and_shape = v->GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_and_shape.GetShape();

#if ORT_API_VERSION >= 12
  auto memory_info = v->GetTensorMemoryInfo();
#else
  const OrtMemoryInfo *mem_info = nullptr;
  OrtStatus *status = Ort::GetApi().GetTensorMemoryInfo(value, &mem_info);
  if (status != nullptr) {
    const char *msg = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    SHERPA_ONNX_LOGE("Failed to get tensor memory info with error: '%s'", msg);
    SHERPA_ONNX_EXIT(-1);
  }
#endif

  switch (type_and_shape.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<int32_t>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<int64_t>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<float>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<uint16_t>(),
          type_and_shape.GetElementCount() * sizeof(uint16_t), shape.data(),
          shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<uint16_t>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return Ort::Value::CreateTensor(
          memory_info, v->GetTensorMutableData<bool>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    default:
      SHERPA_ONNX_LOGE("Unsupported type: %d\n",
                       static_cast<int32_t>(type_and_shape.GetElementType()));
      SHERPA_ONNX_EXIT(-1);
      // unreachable code
      return Ort::Value{nullptr};
  }
}

float ComputeSum(const Ort::Value *v, int32_t n /*= -1*/) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  auto size = static_cast<int32_t>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));
  if (n != -1 && n < size && n > 0) {
    size = n;
  }

  const float *p = v->GetTensorData<float>();

  return std::accumulate(p, p + size, 1.0f);
}

float ComputeMean(const Ort::Value *v, int32_t n /*= -1*/) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  auto size = static_cast<int32_t>(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()));

  if (n != -1 && n < size && n > 0) {
    size = n;
  }

  auto sum = ComputeSum(v, n);
  return sum / size;
}

void PrintShape(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  std::ostringstream os;
  for (auto i : shape) {
    os << i << ", ";
  }
  os << "\n";
  SHERPA_ONNX_LOGE("%s", os.str().c_str());
}

template <typename T /*= float*/>
void Print1D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const T *d = v->GetTensorData<T>();
  std::ostringstream os;
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    os << d[i] << " ";
  }
  os << "\n";
  SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
}

template void Print1D<int64_t>(const Ort::Value *v);
template void Print1D<int32_t>(const Ort::Value *v);
template void Print1D<float>(const Ort::Value *v);

template <typename T /*= float*/>
void Print2D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const T *d = v->GetTensorData<T>();

  std::ostringstream os;
  for (int32_t r = 0; r != static_cast<int32_t>(shape[0]); ++r) {
    for (int32_t c = 0; c != static_cast<int32_t>(shape[1]); ++c, ++d) {
      os << *d << " ";
    }
    os << "\n";
  }
  SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
}

template void Print2D<int64_t>(const Ort::Value *v);
template void Print2D<float>(const Ort::Value *v);

void Print3D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const float *d = v->GetTensorData<float>();

  for (int32_t p = 0; p != static_cast<int32_t>(shape[0]); ++p) {
    SHERPA_ONNX_LOGE("---plane %d---\n", p);
    for (int32_t r = 0; r != static_cast<int32_t>(shape[1]); ++r) {
      for (int32_t c = 0; c != static_cast<int32_t>(shape[2]); ++c, ++d) {
        SHERPA_ONNX_LOGE("%.3f ", *d);
      }
      SHERPA_ONNX_LOGE("\n");
    }
  }
  SHERPA_ONNX_LOGE("\n");
}

void Print4D(const Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const float *d = v->GetTensorData<float>();

  for (int32_t p = 0; p != static_cast<int32_t>(shape[0]); ++p) {
    SHERPA_ONNX_LOGE("---plane %d---\n", p);
    for (int32_t q = 0; q != static_cast<int32_t>(shape[1]); ++q) {
      SHERPA_ONNX_LOGE("---subplane %d---\n", q);
      for (int32_t r = 0; r != static_cast<int32_t>(shape[2]); ++r) {
        for (int32_t c = 0; c != static_cast<int32_t>(shape[3]); ++c, ++d) {
          SHERPA_ONNX_LOGE("%.3f ", *d);
        }
        SHERPA_ONNX_LOGE("\n");
      }
      SHERPA_ONNX_LOGE("\n");
    }
  }
  SHERPA_ONNX_LOGE("\n");
}

Ort::Value Repeat(OrtAllocator *allocator, Ort::Value *cur_encoder_out,
                  const std::vector<int32_t> &hyps_num_split) {
  std::vector<int64_t> cur_encoder_out_shape =
      cur_encoder_out->GetTensorTypeAndShapeInfo().GetShape();

  std::array<int64_t, 2> ans_shape{hyps_num_split.back(),
                                   cur_encoder_out_shape[1]};

  Ort::Value ans = Ort::Value::CreateTensor<float>(allocator, ans_shape.data(),
                                                   ans_shape.size());

  const float *src = cur_encoder_out->GetTensorData<float>();
  float *dst = ans.GetTensorMutableData<float>();
  int32_t batch_size = static_cast<int32_t>(hyps_num_split.size()) - 1;
  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t cur_stream_hyps_num = hyps_num_split[b + 1] - hyps_num_split[b];
    for (int32_t i = 0; i != cur_stream_hyps_num; ++i) {
      std::copy(src, src + cur_encoder_out_shape[1], dst);
      dst += cur_encoder_out_shape[1];
    }
    src += cur_encoder_out_shape[1];
  }
  return ans;
}

CopyableOrtValue::CopyableOrtValue(const CopyableOrtValue &other) {
  *this = other;
}

CopyableOrtValue &CopyableOrtValue::operator=(const CopyableOrtValue &other) {
  if (this == &other) {
    return *this;
  }
  if (other.value) {
    Ort::AllocatorWithDefaultOptions allocator;
    value = Clone(allocator, &other.value);
  }
  return *this;
}

CopyableOrtValue::CopyableOrtValue(CopyableOrtValue &&other) noexcept {
  *this = std::move(other);
}

CopyableOrtValue &CopyableOrtValue::operator=(
    CopyableOrtValue &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  value = std::move(other.value);
  return *this;
}

std::vector<CopyableOrtValue> Convert(std::vector<Ort::Value> values) {
  std::vector<CopyableOrtValue> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v));
  }

  return ans;
}

std::vector<Ort::Value> Convert(std::vector<CopyableOrtValue> values) {
  std::vector<Ort::Value> ans;
  ans.reserve(values.size());

  for (auto &v : values) {
    ans.emplace_back(std::move(v.value));
  }

  return ans;
}

std::string LookupCustomModelMetaData(const Ort::ModelMetadata &meta_data,
                                      const char *key,
                                      OrtAllocator *allocator) {
// Note(fangjun): We only tested 1.17.1 and 1.11.0
// For other versions, we may need to change it
#if ORT_API_VERSION >= 12
  auto v = meta_data.LookupCustomMetadataMapAllocated(key, allocator);
  return v ? v.get() : "";
#else
  auto v = meta_data.LookupCustomMetadataMap(key, allocator);
  std::string ans = v ? v : "";
  allocator->Free(allocator, v);
  return ans;
#endif
}

// Convert IEEE 754 half-precision (16-bit) float to single-precision (32-bit)
// float. Handles special cases: zero, subnormal, normal, infinity, and NaN.
float HalfBitsToFloat(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  const uint32_t exp = (h & 0x7C00u) >> 10;
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
uint16_t FloatToHalfBits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu);
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
    uint32_t half_m = mant >> 13;
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

}  // namespace sherpa_onnx
