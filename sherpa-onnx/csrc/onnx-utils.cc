// sherpa/csrc/onnx-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/onnx-utils.h"

#include <string>
#include <vector>

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

}  // namespace sherpa_onnx
