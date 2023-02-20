// sherpa-onnx/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer-transducer-model.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    const OnlineTransducerModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_{},
      allocator_{} {
  sess_opts_.SetIntraOpNumThreads(config.num_threads);
  sess_opts_.SetInterOpNumThreads(config.num_threads);

  InitEncoder(config.encoder_filename);
  InitDecoder(config.decoder_filename);
  InitJoiner(config.joiner_filename);

  fprintf(stderr, "here in zipformer\n");
  exit(-1);
}

void OnlineZipformerTransducerModel::InitEncoder(const std::string &filename) {
  encoder_sess_ = std::make_unique<Ort::Session>(
      env_, SHERPA_MAYBE_WIDE(filename).c_str(), sess_opts_);

  GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                &encoder_input_names_ptr_);

  GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                 &encoder_output_names_ptr_);

  // get meta data
  Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
  if (config_.debug) {
    std::ostringstream os;
    os << "---encoder---\n";
    PrintModelMetadata(os, meta_data);
    fprintf(stderr, "%s\n", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  SHERPA_ONNX_READ_META_DATA_VEC(encoder_dims_, "encoder_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(attention_dims_, "attention_dims");
  SHERPA_ONNX_READ_META_DATA_VEC(num_encoder_layers_, "num_encoder_layers");
  SHERPA_ONNX_READ_META_DATA_VEC(cnn_module_kernels_, "cnn_module_kernels");
  SHERPA_ONNX_READ_META_DATA_VEC(left_context_len_, "left_context_len");

  SHERPA_ONNX_READ_META_DATA(T_, "T");
  SHERPA_ONNX_READ_META_DATA(decode_chunk_len_, "decode_chunk_len");

  if (config_.debug) {
    auto print = [](const std::vector<int32_t> &v, const char *name) {
      fprintf(stderr, "%s: ", name);
      for (auto i : v) {
        fprintf(stderr, "%d ", i);
      }
      fprintf(stderr, "\n");
    };
    print(encoder_dims_, "encoder_dims");
    print(attention_dims_, "attention_dims");
    print(num_encoder_layers_, "num_encoder_layers");
    print(cnn_module_kernels_, "cnn_module_kernels");
    print(left_context_len_, "left_context_len");
    fprintf(stderr, "T: %d\n", T_);
    fprintf(stderr, "decode_chunk_len_: %d\n", decode_chunk_len_);
  }
}

void OnlineZipformerTransducerModel::InitDecoder(const std::string &filename) {}
void OnlineZipformerTransducerModel::InitJoiner(const std::string &filename) {}

std::vector<Ort::Value> OnlineZipformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  return {};
}

std::vector<std::vector<Ort::Value>>
OnlineZipformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  return {};
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::GetEncoderInitStates() {
  return {};
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineZipformerTransducerModel::RunEncoder(Ort::Value features,
                                           std::vector<Ort::Value> states) {
  Ort::Value a{nullptr};
  std::vector<Ort::Value> b;
  return {std::move(a), std::move(b)};
}

Ort::Value OnlineZipformerTransducerModel::BuildDecoderInput(
    const std::vector<OnlineTransducerDecoderResult> &results) {
  return Ort::Value{nullptr};
}

Ort::Value OnlineZipformerTransducerModel::RunDecoder(
    Ort::Value decoder_input) {
  return Ort::Value{nullptr};
}

Ort::Value OnlineZipformerTransducerModel::RunJoiner(Ort::Value encoder_out,
                                                     Ort::Value decoder_out) {
  return Ort::Value{nullptr};
}

}  // namespace sherpa_onnx
