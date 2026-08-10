// sherpa-onnx/csrc/offline-cohere-transcribe-model.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-cohere-transcribe-model.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
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

static inline bool IsCudaProvider(const std::string &provider) {
  return Contains(provider, "cuda");
}

}  // namespace

class OfflineCohereTranscribeModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.cohere_transcribe.encoder),
        sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.cohere_transcribe.decoder),
        sess_opts_);
    InitDecoder(nullptr, 0);

    InitCudaIOBinding();

    InitState();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    SHERPA_ONNX_LOGE(
        "Please copy files to SD card for Cohere Transcribe. It does not "
        "support using a manager");
    SHERPA_ONNX_EXIT(-1);
  }

  std::pair<Ort::Value, Ort::Value> ForwardEncoder(Ort::Value features) {
    std::vector<Ort::Value> encoder_out;

    if (use_cuda_iobinding_) {
      // Encoder outputs are n_layer_cross_k and n_layer_cross_v, which are used
      // multiple times in decoder steps. Keep them on GPU to avoid
      // device<->host copies.
      Ort::IoBinding binding(*encoder_sess_);
      binding.BindInput(encoder_input_names_ptr_[0], features);

      binding.BindOutput(encoder_output_names_ptr_[0], *cuda_mem_info_);
      binding.BindOutput(encoder_output_names_ptr_[1], *cuda_mem_info_);

      binding.SynchronizeInputs();
      encoder_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      encoder_out = binding.GetOutputValues();
    } else {
      encoder_out = encoder_sess_->Run(
          {}, encoder_input_names_ptr_.data(), &features, 1,
          encoder_output_names_ptr_.data(), encoder_output_names_ptr_.size());
    }

    return {std::move(encoder_out[0]), std::move(encoder_out[1])};
  }

  std::tuple<Ort::Value, Ort::Value, Ort::Value> ForwardDecoder(
      Ort::Value tokens, Ort::Value n_layer_self_k_cache,
      Ort::Value n_layer_self_v_cache, Ort::Value n_layer_cross_k,
      Ort::Value n_layer_cross_v, Ort::Value offset) {
    std::array<Ort::Value, 6> decoder_input = {std::move(tokens),
                                               std::move(n_layer_self_k_cache),
                                               std::move(n_layer_self_v_cache),
                                               std::move(n_layer_cross_k),
                                               std::move(n_layer_cross_v),
                                               std::move(offset)};

    std::vector<Ort::Value> decoder_out;

    if (use_cuda_iobinding_) {
      // CPU-side sampling needs logits on CPU, while self KV cache should
      // remain on GPU to avoid large device<->host copies between decode steps.
      Ort::IoBinding binding(*decoder_sess_);
      for (size_t i = 0; i < decoder_input.size(); ++i) {
        binding.BindInput(decoder_input_names_ptr_[i], decoder_input[i]);
      }

      binding.BindOutput(decoder_output_names_ptr_[0], cpu_mem_info_);
      binding.BindOutput(decoder_output_names_ptr_[1], *cuda_mem_info_);
      binding.BindOutput(decoder_output_names_ptr_[2], *cuda_mem_info_);

      binding.SynchronizeInputs();
      decoder_sess_->Run(Ort::RunOptions{nullptr}, binding);
      binding.SynchronizeOutputs();
      decoder_out = binding.GetOutputValues();
    } else {
      decoder_out = decoder_sess_->Run(
          {}, decoder_input_names_ptr_.data(), decoder_input.data(),
          decoder_input.size(), decoder_output_names_ptr_.data(),
          decoder_output_names_ptr_.size());
    }

    return std::tuple<Ort::Value, Ort::Value, Ort::Value>{
        std::move(decoder_out[0]), std::move(decoder_out[1]),
        std::move(decoder_out[2])};
  }

  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache() {
    return {View(&n_layer_self_k_cache_), View(&n_layer_self_v_cache_)};
  }

  OrtAllocator *Allocator() { return allocator_; }

  int32_t GetMaxSeqLen() const { return max_seq_len_; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      encoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!encoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize encoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

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
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(model_type, "model_type");
    if (!Contains(model_type, "cohere-transcribe")) {
      SHERPA_ONNX_LOGE(
          "Expect model type 'cohere-transcribe-03-2026'. Given: '%s'",
          model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    auto shape = encoder_sess_->GetOutputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();

    if (shape.size() != 4) {
      SHERPA_ONNX_LOGE("Expect 4-d for encoder output 0. Given: %d",
                       static_cast<int32_t>(shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    num_layers_ = shape[0];
    hidden_size_ = shape[3];
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    if (model_data) {
      decoder_sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!decoder_sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize decoder session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(decoder_sess_.get(), &decoder_input_names_,
                  &decoder_input_names_ptr_);

    GetOutputNames(decoder_sess_.get(), &decoder_output_names_,
                   &decoder_output_names_ptr_);

    vocab_size_ = decoder_sess_->GetOutputTypeInfo(0)
                      .GetTensorTypeAndShapeInfo()
                      .GetShape()
                      .back();

    auto shape = decoder_sess_->GetOutputTypeInfo(1)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();

    if (shape.size() != 5) {
      SHERPA_ONNX_LOGE("Expect 5-d for decoder output 1. Given: %d",
                       static_cast<int32_t>(shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (shape[0] != num_layers_) {
      SHERPA_ONNX_LOGE("Expected num_layers %d. Given: %d", num_layers_,
                       static_cast<int32_t>(shape[0]));
      SHERPA_ONNX_EXIT(-1);
    }

    num_heads_ = shape[2];
    max_seq_len_ = shape[3];
    head_dim_ = hidden_size_ / num_heads_;

    if (config_.debug) {
      std::ostringstream os;
      os << "num_layers: " << num_layers_ << "\n";
      os << "max_seq_len: " << max_seq_len_ << "\n";
      os << "num_heads: " << num_heads_ << "\n";
      os << "hidden_size: " << hidden_size_ << "\n";
      os << "head_dim: " << head_dim_ << "\n";
      os << "vocab_size: " << vocab_size_ << "\n";
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }
  }

  void InitCudaIOBinding() {
    use_cuda_iobinding_ =
        (!is_cpu_provider_ && IsCudaProvider(config_.provider));
    if (use_cuda_iobinding_) {
      // Use device 0 by default. SessionOptions() in sherpa-onnx usually
      // configures the CUDA EP device; binding here only affects output memory.
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
  }

  void InitState() {
    std::array<int64_t, 5> shape{num_layers_, 1, num_heads_, max_seq_len_,
                                 head_dim_};

    n_layer_self_k_cache_ = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    n_layer_self_v_cache_ = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    auto n = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];

    float *p_k = n_layer_self_k_cache_.GetTensorMutableData<float>();
    float *p_v = n_layer_self_v_cache_.GetTensorMutableData<float>();

    memset(p_k, 0, sizeof(float) * n);
    memset(p_v, 0, sizeof(float) * n);
  }

  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  Ort::MemoryInfo cpu_mem_info_;
  std::unique_ptr<Ort::MemoryInfo> cuda_mem_info_;
  bool use_cuda_iobinding_ = false;
  bool is_cpu_provider_ = false;

  std::unique_ptr<Ort::Session> encoder_sess_;
  std::unique_ptr<Ort::Session> decoder_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  int32_t num_layers_ = 0;
  int32_t max_seq_len_ = 0;
  int32_t num_heads_ = 0;
  int32_t hidden_size_ = 0;
  int32_t head_dim_ = 0;
  int32_t vocab_size_ = 0;

  Ort::Value n_layer_self_k_cache_{nullptr};
  Ort::Value n_layer_self_v_cache_{nullptr};
};

OfflineCohereTranscribeModel::OfflineCohereTranscribeModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineCohereTranscribeModel::OfflineCohereTranscribeModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineCohereTranscribeModel::~OfflineCohereTranscribeModel() = default;

std::pair<Ort::Value, Ort::Value> OfflineCohereTranscribeModel::ForwardEncoder(
    Ort::Value features) const {
  return impl_->ForwardEncoder(std::move(features));
}

std::tuple<Ort::Value, Ort::Value, Ort::Value>
OfflineCohereTranscribeModel::ForwardDecoder(Ort::Value tokens,
                                             Ort::Value n_layer_self_k_cache,
                                             Ort::Value n_layer_self_v_cache,
                                             Ort::Value n_layer_cross_k,
                                             Ort::Value n_layer_cross_v,
                                             Ort::Value offset) const {
  return impl_->ForwardDecoder(
      std::move(tokens), std::move(n_layer_self_k_cache),
      std::move(n_layer_self_v_cache), std::move(n_layer_cross_k),
      std::move(n_layer_cross_v), std::move(offset));
}

std::pair<Ort::Value, Ort::Value>
OfflineCohereTranscribeModel::GetInitialSelfKVCache() const {
  return impl_->GetInitialSelfKVCache();
}

OrtAllocator *OfflineCohereTranscribeModel::Allocator() const {
  return impl_->Allocator();
}

int32_t OfflineCohereTranscribeModel::GetMaxSeqLen() const {
  return impl_->GetMaxSeqLen();
}

#if __ANDROID_API__ >= 9
template OfflineCohereTranscribeModel::OfflineCohereTranscribeModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineCohereTranscribeModel::OfflineCohereTranscribeModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx
