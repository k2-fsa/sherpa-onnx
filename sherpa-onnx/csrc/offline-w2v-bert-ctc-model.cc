// sherpa-onnx/csrc/offline-w2v-bert-ctc-model.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-model.h"

#include <array>
#include <memory>
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

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Normalizes each of the feat_dim bins of features (num_frames rows) to zero
// mean and unit variance over the utterance, as transformers'
// SeamlessM4TFeatureExtractor does.
void NormalizeW2vBertFeatures(float *features, int32_t num_frames,
                              int32_t feat_dim) {
  if (num_frames < 2) return;
  using RowMajorMat =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<RowMajorMat> x(features, num_frames, feat_dim);

  Eigen::RowVectorXf mean = x.colwise().mean();
  x.rowwise() -= mean;

  // The sample variance, as the feature extractor computes it.
  Eigen::Array<float, 1, Eigen::Dynamic> var =
      x.array().square().colwise().sum() / static_cast<float>(num_frames - 1);
  Eigen::Array<float, 1, Eigen::Dynamic> inv_std = (var + 1e-7f).rsqrt();
  x.array().rowwise() *= inv_std;
}

}  // namespace

class OfflineW2vBertCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    // From the path, not a buffer: the fp32 weights are in model.weights
    // beside model.onnx.
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.w2v_bert_ctc.model), sess_opts_);
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.w2v_bert_ctc.model);
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    Init();
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    // The model masks the padding, whatever its value. With an int8 model
    // the batch still shares activation quantization ranges, so a batch can
    // differ slightly from single decoding; see the class comment.
    std::array<Ort::Value, 2> inputs = {std::move(features),
                                        std::move(features_length)};
    // The graph's outputs are log-softmax normalized (Init() checks the
    // export version).
    return sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
  }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t VocabSize() const { return vocab_size_; }
  OrtAllocator *Allocator() { return allocator_; }

 private:
  void Init() {
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
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

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
    if (model_type != "w2v-bert-ctc") {
      SHERPA_ONNX_LOGE(
          "Expected model type 'w2v-bert-ctc'. Given: '%s'. Please export the "
          "model with scripts/w2v-bert/export_onnx.py",
          model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    // Version 2 normalizes in the graph; version 1 did not. Outputs are
    // taken by position, so either version 2 output naming (log_probs, or
    // the first exports' logits) loads.
    int32_t version = 0;
    SHERPA_ONNX_READ_META_DATA(version, "version");
    if (version < 2) {
      SHERPA_ONNX_LOGE(
          "This w2v-BERT CTC model has export version %d; version 2 or later "
          "is required. Please export it again with "
          "scripts/w2v-bert/export_onnx.py",
          version);
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
    if (input_names_.size() != 2 || output_names_.size() != 2) {
      SHERPA_ONNX_LOGE(
          "A w2v-BERT CTC model takes input_features and input_lengths and "
          "returns log_probs and log_probs_lengths. This one has %d inputs and "
          "%d outputs. Please export it with scripts/w2v-bert/export_onnx.py",
          static_cast<int32_t>(input_names_.size()),
          static_cast<int32_t>(output_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(vocab_size_, "vocab_size");
    SHERPA_ONNX_READ_META_DATA(subsampling_factor_, "subsampling_factor");
  }

  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t vocab_size_ = 0;
  int32_t subsampling_factor_ = 0;
};

OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineW2vBertCtcModel::~OfflineW2vBertCtcModel() = default;

std::vector<Ort::Value> OfflineW2vBertCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineW2vBertCtcModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineW2vBertCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineW2vBertCtcModel::Allocator() const {
  return impl_->Allocator();
}

void OfflineW2vBertCtcModel::NormalizeFeatures(float *features,
                                               int32_t num_frames,
                                               int32_t feat_dim) const {
  NormalizeW2vBertFeatures(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
