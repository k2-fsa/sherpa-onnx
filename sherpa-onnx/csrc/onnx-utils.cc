// sherpa-onnx/csrc/onnx-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/onnx-utils.h"

#include <fstream>
#include <string>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#include "android/log.h"
#endif

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

void GetInputNames(Ort::Session *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t node_count = sess->GetInputCount();
  input_names->resize(node_count);
  input_names_ptr->resize(node_count);
  for (size_t i = 0; i != node_count; ++i) {
    auto tmp = sess->GetInputNameAllocated(i, allocator);
    (*input_names)[i] = tmp.get();
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
    auto tmp = sess->GetOutputNameAllocated(i, allocator);
    (*output_names)[i] = tmp.get();
    (*output_names_ptr)[i] = (*output_names)[i].c_str();
  }
}

void PrintModelMetadata(std::ostream &os, const Ort::ModelMetadata &meta_data) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::AllocatedStringPtr> v =
      meta_data.GetCustomMetadataMapKeysAllocated(allocator);
  for (const auto &key : v) {
    auto p = meta_data.LookupCustomMetadataMapAllocated(key.get(), allocator);
    os << key.get() << "=" << p.get() << "\n";
  }
}

Ort::Value Clone(const Ort::Value *v) {
  auto type_and_shape = v->GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_and_shape.GetShape();

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  switch (type_and_shape.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return Ort::Value::CreateTensor(
          memory_info,
          const_cast<Ort::Value *>(v)->GetTensorMutableData<int32_t>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return Ort::Value::CreateTensor(
          memory_info,
          const_cast<Ort::Value *>(v)->GetTensorMutableData<int64_t>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return Ort::Value::CreateTensor(
          memory_info,
          const_cast<Ort::Value *>(v)->GetTensorMutableData<float>(),
          type_and_shape.GetElementCount(), shape.data(), shape.size());
    default:
      fprintf(stderr, "Unsupported type: %d\n",
              static_cast<int32_t>(type_and_shape.GetElementType()));
      exit(-1);
      // unreachable code
      return Ort::Value{nullptr};
  }
}

void Print1D(Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const float *d = v->GetTensorData<float>();
  for (int32_t i = 0; i != static_cast<int32_t>(shape[0]); ++i) {
    fprintf(stderr, "%.3f ", d[i]);
  }
  fprintf(stderr, "\n");
}

void Print2D(Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const float *d = v->GetTensorData<float>();

  for (int32_t r = 0; r != static_cast<int32_t>(shape[0]); ++r) {
    for (int32_t c = 0; c != static_cast<int32_t>(shape[1]); ++c, ++d) {
      fprintf(stderr, "%.3f ", *d);
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

void Print3D(Ort::Value *v) {
  std::vector<int64_t> shape = v->GetTensorTypeAndShapeInfo().GetShape();
  const float *d = v->GetTensorData<float>();

  for (int32_t p = 0; p != static_cast<int32_t>(shape[0]); ++p) {
    fprintf(stderr, "---plane %d---\n", p);
    for (int32_t r = 0; r != static_cast<int32_t>(shape[1]); ++r) {
      for (int32_t c = 0; c != static_cast<int32_t>(shape[2]); ++c, ++d) {
        fprintf(stderr, "%.3f ", *d);
      }
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");
}

std::vector<char> ReadFile(const std::string &filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
  return buffer;
}

#if __ANDROID_API__ >= 9
std::vector<char> ReadFile(AAssetManager *mgr, const std::string &filename) {
  AAsset *asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
  if (!asset) {
    __android_log_print(ANDROID_LOG_FATAL, "sherpa-onnx",
                        "Read binary file: Load %s failed", filename.c_str());
    exit(-1);
  }

  auto p = reinterpret_cast<const char *>(AAsset_getBuffer(asset));
  size_t asset_length = AAsset_getLength(asset);

  std::vector<char> buffer(p, p + asset_length);
  AAsset_close(asset);

  return buffer;
}
#endif

}  // namespace sherpa_onnx
