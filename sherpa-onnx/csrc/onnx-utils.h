// sherpa-onnx/csrc/onnx-utils.h
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo
#ifndef SHERPA_ONNX_CSRC_ONNX_UTILS_H_
#define SHERPA_ONNX_CSRC_ONNX_UTILS_H_

#ifdef _MSC_VER
// For ToWide() below
#include <codecvt>
#include <locale>
#endif

#include <cassert>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

/**
 * Get the input names of a model.
 *
 * @param sess An onnxruntime session.
 * @param input_names. On return, it contains the input names of the model.
 * @param input_names_ptr. On return, input_names_ptr[i] contains
 *                         input_names[i].c_str()
 */
void GetInputNames(Ort::Session *sess, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr);

/**
 * Get the output names of a model.
 *
 * @param sess An onnxruntime session.
 * @param output_names. On return, it contains the output names of the model.
 * @param output_names_ptr. On return, output_names_ptr[i] contains
 *                         output_names[i].c_str()
 */
void GetOutputNames(Ort::Session *sess, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr);

/**
 * Get the output frame of Encoder
 *
 * @param allocator allocator of onnxruntime
 * @param encoder_out encoder out tensor
 * @param t frame_index
 *
 */
Ort::Value GetEncoderOutFrame(OrtAllocator *allocator, Ort::Value *encoder_out,
                              int32_t t);

std::string LookupCustomModelMetaData(const Ort::ModelMetadata &meta_data,
                                      const char *key, OrtAllocator *allocator);

void PrintModelMetadata(std::ostream &os,
                        const Ort::ModelMetadata &meta_data);  // NOLINT

// Return a deep copy of v
Ort::Value Clone(OrtAllocator *allocator, const Ort::Value *v);

// Return a shallow copy
Ort::Value View(Ort::Value *v);

float ComputeSum(const Ort::Value *v, int32_t n = -1);
float ComputeMean(const Ort::Value *v, int32_t n = -1);

// Print a 1-D tensor to stderr
template <typename T = float>
void Print1D(const Ort::Value *v);

// Print a 2-D tensor to stderr
template <typename T = float>
void Print2D(const Ort::Value *v);

// Print a 3-D tensor to stderr
void Print3D(const Ort::Value *v);

// Print a 4-D tensor to stderr
void Print4D(const Ort::Value *v);

void PrintShape(const Ort::Value *v);

template <typename T = float>
void Fill(Ort::Value *tensor, T value) {
  auto n = tensor->GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
  auto p = tensor->GetTensorMutableData<T>();
  std::fill(p, p + n, value);
}

// TODO(fangjun): Document it
Ort::Value Repeat(OrtAllocator *allocator, Ort::Value *cur_encoder_out,
                  const std::vector<int32_t> &hyps_num_split);

struct CopyableOrtValue {
  Ort::Value value{nullptr};

  CopyableOrtValue() = default;

  /*explicit*/ CopyableOrtValue(Ort::Value v)  // NOLINT
      : value(std::move(v)) {}

  CopyableOrtValue(const CopyableOrtValue &other);

  CopyableOrtValue &operator=(const CopyableOrtValue &other);

  CopyableOrtValue(CopyableOrtValue &&other) noexcept;

  CopyableOrtValue &operator=(CopyableOrtValue &&other) noexcept;
};

std::vector<CopyableOrtValue> Convert(std::vector<Ort::Value> values);

std::vector<Ort::Value> Convert(std::vector<CopyableOrtValue> values);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONNX_UTILS_H_
