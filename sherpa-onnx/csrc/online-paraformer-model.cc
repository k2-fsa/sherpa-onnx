// sherpa-onnx/csrc/online-paraformer-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-paraformer-model.h"

#include <algorithm>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OnlineParaformerModel::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value> ForwardEncoder(
      Ort::Value features, Ort::Value features_length) {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

  std::vector<Ort::Value> ForwardDecoder(Ort::Value encoder_out,
                                         Ort::Value encoder_out_length,
                                         Ort::Value acoustic_embedding,
                                         Ort::Value acoustic_embedding_length,
                                         std::vector<Ort::Value> states) {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

  int32_t VocabSize() const { return vocab_size_; }

  int32_t LfrWindowSize() const { return lfr_window_size_; }

  int32_t LfrWindowShift() const { return lfr_window_shift_; }

  const std::vector<float> &NegativeMean() const { return neg_mean_; }

  const std::vector<float> &InverseStdDev() const { return inv_stddev_; }

  OrtAllocator *Allocator() const { return allocator_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(encoder_sess_.get(), &encoder_input_names_,
                  &encoder_input_names_ptr_);

    GetOutputNames(encoder_sess_.get(), &encoder_output_names_,
                   &encoder_output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = encoder_sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(lfr_window_size_, "lfr_window_size");
    SHERPA_ONNX_READ_META_DATA(lfr_window_shift_, "lfr_window_shift");
    SHERPA_ONNX_READ_META_DATA(encoder_output_size_, "encoder_output_size");
    SHERPA_ONNX_READ_META_DATA(decoder_num_blocks_, "decoder_num_blocks");
    SHERPA_ONNX_READ_META_DATA(decoder_kernel_size_, "decoder_kernel_size");

    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(neg_mean_, "neg_mean");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(inv_stddev_, "inv_stddev");
  }

 private:
  OnlineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> encoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<float> neg_mean_;
  std::vector<float> inv_stddev_;

  int32_t vocab_size_ = 0;  // initialized in Init
  int32_t lfr_window_size_ = 0;
  int32_t lfr_window_shift_ = 0;

  int32_t encoder_output_size_ = 0;
  int32_t decoder_num_blocks_ = 0;
  int32_t decoder_kernel_size_ = 0;
};

OnlineParaformerModel::OnlineParaformerModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineParaformerModel::~OnlineParaformerModel() = default;

std::tuple<Ort::Value, Ort::Value, Ort::Value>
OnlineParaformerModel::ForwardEncoder(Ort::Value features,
                                      Ort::Value features_length) {
  return impl_->ForwardEncoder(std::move(features), std::move(features_length));
}

std::vector<Ort::Value> OnlineParaformerModel::ForwardDecoder(
    Ort::Value encoder_out, Ort::Value encoder_out_length,
    Ort::Value acoustic_embedding, Ort::Value acoustic_embedding_length,
    std::vector<Ort::Value> states) {
  return impl_->ForwardDecoder(
      std::move(encoder_out), std::move(encoder_out_length),
      std::move(acoustic_embedding), std::move(acoustic_embedding_length),
      std::move(states));
}

int32_t OnlineParaformerModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OnlineParaformerModel::LfrWindowSize() const {
  return impl_->LfrWindowSize();
}
int32_t OnlineParaformerModel::LfrWindowShift() const {
  return impl_->LfrWindowShift();
}
const std::vector<float> &OnlineParaformerModel::NegativeMean() const {
  return impl_->NegativeMean();
}
const std::vector<float> &OnlineParaformerModel::InverseStdDev() const {
  return impl_->InverseStdDev();
}

OrtAllocator *OnlineParaformerModel::Allocator() const {
  return impl_->Allocator();
}

}  // namespace sherpa_onnx
