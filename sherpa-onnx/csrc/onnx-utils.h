// sherpa-onnx/csrc/onnx-utils.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONNX_UTILS_H_
#define SHERPA_ONNX_CSRC_ONNX_UTILS_H_

#ifdef _MSC_VER
// For ToWide() below
#include <codecvt>
#include <locale>
#endif

#include <ostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

#ifdef _MSC_VER
// See
// https://stackoverflow.com/questions/2573834/c-convert-string-or-char-to-wstring-or-wchar-t
static std::wstring ToWide(const std::string &s) {
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  return converter.from_bytes(s);
}
#define SHERPA_MAYBE_WIDE(s) ToWide(s)
#else
#define SHERPA_MAYBE_WIDE(s) s
#endif

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

void PrintModelMetadata(std::ostream &os,
                        const Ort::ModelMetadata &meta_data);  // NOLINT

// Return a shallow copy of v
Ort::Value Clone(const Ort::Value *v);

// Print a 1-D tensor to stderr
void Print1D(Ort::Value *v);

// Print a 2-D tensor to stderr
void Print2D(Ort::Value *v);

// Print a 3-D tensor to stderr
void Print3D(Ort::Value *v);

template <typename T = float>
void Fill(Ort::Value *tensor, T value) {
  auto n = tensor->GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
  auto p = tensor->GetTensorMutableData<T>();
  std::fill(p, p + n, value);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONNX_UTILS_H_
