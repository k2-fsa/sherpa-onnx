// sherpa-onnx/csrc/offline-ct-transformer-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ct-transformer-model.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineCtTransformerModel::Impl {
 public:
  explicit Impl(const OfflinePunctuationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config_.ct_transformer);
    Init(buf.data(), buf.size());
  }

#if __ANDROID_API__ >= 9
  Impl(AAssetManager *mgr, const OfflinePunctuationModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.ct_transformer);
    Init(buf.data(), buf.size());
  }
#endif

  Ort::Value Forward(Ort::Value text, Ort::Value text_len) {
    std::array<Ort::Value, 2> inputs = {std::move(text), std::move(text_len)};

    auto ans =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    return std::move(ans[0]);
  }

  OrtAllocator *Allocator() { return allocator_; }

  const OfflineCtTransformerModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::vector<std::string> tokens;
    SHERPA_ONNX_READ_META_DATA_VEC_STRING_SEP(tokens, "tokens", "|");

    int32_t vocab_size = 0;
    SHERPA_ONNX_READ_META_DATA(vocab_size, "vocab_size");
    if (static_cast<int32_t>(tokens.size()) != vocab_size) {
      SHERPA_ONNX_LOGE("tokens.size() %d != vocab_size %d",
                       static_cast<int32_t>(tokens.size()), vocab_size);
      exit(-1);
    }

    SHERPA_ONNX_READ_META_DATA_VEC_STRING_SEP(meta_data_.id2punct,
                                              "punctuations", "|");

    std::string unk_symbol;
    SHERPA_ONNX_READ_META_DATA_STR(unk_symbol, "unk_symbol");

    // output shape is (N, T, num_punctuations)
    meta_data_.num_punctuations =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];

    int32_t i = 0;
    for (const auto &t : tokens) {
      meta_data_.token2id[t] = i;
      i += 1;
    }

    i = 0;
    for (const auto &p : meta_data_.id2punct) {
      meta_data_.punct2id[p] = i;
      i += 1;
    }

    meta_data_.unk_id = meta_data_.token2id.at(unk_symbol);

    meta_data_.dot_id = meta_data_.punct2id.at("。");
    meta_data_.comma_id = meta_data_.punct2id.at("，");
    meta_data_.quest_id = meta_data_.punct2id.at("？");
    meta_data_.pause_id = meta_data_.punct2id.at("、");
    meta_data_.underline_id = meta_data_.punct2id.at("_");

    if (config_.debug) {
      std::ostringstream os;
      os << "vocab_size: " << meta_data_.token2id.size() << "\n";
      os << "num_punctuations: " << meta_data_.num_punctuations << "\n";
      os << "punctuations: ";
      for (const auto &s : meta_data_.id2punct) {
        os << s << " ";
      }
      os << "\n";
      SHERPA_ONNX_LOGE("\n%s\n", os.str().c_str());
    }
  }

 private:
  OfflinePunctuationModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OfflineCtTransformerModelMetaData meta_data_;
};

OfflineCtTransformerModel::OfflineCtTransformerModel(
    const OfflinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

#if __ANDROID_API__ >= 9
OfflineCtTransformerModel::OfflineCtTransformerModel(
    AAssetManager *mgr, const OfflinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}
#endif

OfflineCtTransformerModel::~OfflineCtTransformerModel() = default;

Ort::Value OfflineCtTransformerModel::Forward(Ort::Value text,
                                              Ort::Value text_len) const {
  return impl_->Forward(std::move(text), std::move(text_len));
}

OrtAllocator *OfflineCtTransformerModel::Allocator() const {
  return impl_->Allocator();
}

const OfflineCtTransformerModelMetaData &
OfflineCtTransformerModel::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

}  // namespace sherpa_onnx
