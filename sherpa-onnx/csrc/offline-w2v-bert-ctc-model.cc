// sherpa-onnx/csrc/offline-w2v-bert-ctc-model.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-model.h"

#include <algorithm>
#include <array>
#include <cmath>
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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Two filterbank frames become one input row.
constexpr int32_t kStack = 2;

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
    auto shape = features.GetTensorTypeAndShapeInfo().GetShape();
    auto batch = static_cast<int32_t>(shape[0]);
    auto num_frames = static_cast<int32_t>(shape[1]);
    auto feat_dim = static_cast<int32_t>(shape[2]);
    // An odd last frame is dropped, as the feature extractor drops it.
    int32_t rows = num_frames / kStack;
    size_t row_size = static_cast<size_t>(feat_dim) * kStack;
    size_t utterance_size = rows * row_size;

    // Each utterance's frames are consecutive in memory, so joining two
    // frames into a row only changes the shape, less an odd last frame.
    std::array<int64_t, 3> stacked_shape = {batch, rows,
                                            static_cast<int64_t>(row_size)};
    Ort::Value stacked = Ort::Value::CreateTensor<float>(
        allocator_, stacked_shape.data(), stacked_shape.size());
    const float *p = features.GetTensorData<float>();
    float *q = stacked.GetTensorMutableData<float>();

    std::array<int64_t, 1> length_shape = {batch};
    Ort::Value length = Ort::Value::CreateTensor<int64_t>(
        allocator_, length_shape.data(), length_shape.size());
    int64_t *row_length = length.GetTensorMutableData<int64_t>();
    const int64_t *frames_length = features_length.GetTensorData<int64_t>();

    for (int32_t b = 0; b != batch; ++b) {
      const float *src = p + static_cast<size_t>(b) * num_frames * feat_dim;
      float *dst = q + b * utterance_size;
      row_length[b] = frames_length[b] / kStack;
      size_t valid = row_length[b] * row_size;
      std::copy(src, src + valid, dst);
      // The model masks the padding; zeroing it keeps the recognizer's
      // padding value out of the graph.
      std::fill(dst + valid, dst + utterance_size, 0.0f);
    }

    // With an int8 model the batch still shares activation quantization
    // ranges, so a batch can differ slightly from single decoding; see the
    // class comment.
    std::array<Ort::Value, 2> inputs = {std::move(stacked), std::move(length)};
    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    auto out_shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    LogSoftmax(out[0].GetTensorMutableData<float>(),
               static_cast<int32_t>(out_shape[2]),
               static_cast<int32_t>(out_shape[0] * out_shape[1]));
    return out;
  }

  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t VocabSize() const { return vocab_size_; }
  OrtAllocator *Allocator() { return allocator_; }

 private:
  void Init() {
    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
    if (input_names_.size() != 2 || output_names_.size() != 2) {
      SHERPA_ONNX_LOGE(
          "A w2v-BERT CTC model takes input_features and input_lengths and "
          "returns logits and logits_lengths. This one has %d inputs and %d "
          "outputs. Please export it with scripts/w2v-bert/export_onnx.py",
          static_cast<int32_t>(input_names_.size()),
          static_cast<int32_t>(output_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

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

void NormalizeW2vBertFeatures(float *features, int32_t num_frames,
                              int32_t feat_dim) {
  if (num_frames < 2) return;
  for (int32_t d = 0; d != feat_dim; ++d) {
    double sum = 0;
    for (int32_t t = 0; t != num_frames; ++t) sum += features[t * feat_dim + d];
    double mean = sum / num_frames;
    double squares = 0;
    for (int32_t t = 0; t != num_frames; ++t) {
      double x = features[t * feat_dim + d] - mean;
      squares += x * x;
    }
    // The sample variance, as the feature extractor computes it.
    double scale = 1.0 / std::sqrt(squares / (num_frames - 1) + 1e-7);
    for (int32_t t = 0; t != num_frames; ++t) {
      features[t * feat_dim + d] =
          static_cast<float>((features[t * feat_dim + d] - mean) * scale);
    }
  }
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
