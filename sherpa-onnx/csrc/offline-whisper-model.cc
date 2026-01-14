// sherpa-onnx/csrc/offline-whisper-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-whisper-model.h"

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
  return provider == "cuda";
}

}  // namespace

class OfflineWhisperModel::Impl {
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
        env_, SHERPA_ONNX_TO_ORT_PATH(config.whisper.encoder), sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.whisper.decoder), sess_opts_);
    InitDecoder(nullptr, 0);

    InitCudaIOBinding();
  }

  explicit Impl(const SpokenLanguageIdentificationConfig &config)
      : lid_config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    encoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.whisper.encoder), sess_opts_);
    InitEncoder(nullptr, 0);

    decoder_sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.whisper.decoder), sess_opts_);
    InitDecoder(nullptr, 0);

    InitCudaIOBinding();
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
    {
      auto buf = ReadFile(mgr, config.whisper.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.whisper.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    InitCudaIOBinding();
  }

  template <typename Manager>
  Impl(Manager *mgr, const SpokenLanguageIdentificationConfig &config)
      : lid_config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{},
        cpu_mem_info_(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
        is_cpu_provider_(config.provider == "cpu" || config.provider.empty()) {
    {
      auto buf = ReadFile(mgr, config.whisper.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.whisper.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    InitCudaIOBinding();
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

  std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
             Ort::Value>
  ForwardDecoder(Ort::Value tokens, Ort::Value n_layer_self_k_cache,
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

    return std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value,
                      Ort::Value, Ort::Value>{
        std::move(decoder_out[0]),   std::move(decoder_out[1]),
        std::move(decoder_out[2]),   std::move(decoder_input[3]),
        std::move(decoder_input[4]), std::move(decoder_input[5])};
  }

  int32_t DetectLanguage(Ort::Value &cross_k,    // NOLINT
                         Ort::Value &cross_v) {  // NOLINT
    int64_t token_val = SOT();
    std::array<int64_t, 2> token_shape{1, 1};

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    Ort::Value tokens = Ort::Value::CreateTensor(
        memory_info, &token_val, 1, token_shape.data(), token_shape.size());

    auto self_kv_cache = GetInitialSelfKVCache();

    std::array<int64_t, 1> offset_shape{1};
    Ort::Value offset = Ort::Value::CreateTensor<int64_t>(
        Allocator(), offset_shape.data(), offset_shape.size());
    *(offset.GetTensorMutableData<int64_t>()) = 0;

    auto decoder_out =
        ForwardDecoder(std::move(tokens), std::move(self_kv_cache.first),
                       std::move(self_kv_cache.second), std::move(cross_k),
                       std::move(cross_v), std::move(offset));

    cross_k = std::move(std::get<3>(decoder_out));
    cross_v = std::move(std::get<4>(decoder_out));

    const float *p_logits = std::get<0>(decoder_out).GetTensorData<float>();
    const auto &all_language_ids = GetAllLanguageIDs();

    int32_t lang_id = all_language_ids[0];
    float this_logit = p_logits[lang_id];

    for (int32_t i = 1; i != all_language_ids.size(); ++i) {
      int32_t id = all_language_ids[i];
      float p = p_logits[id];

      if (p > this_logit) {
        this_logit = p;
        lang_id = id;
      }
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("Detected language: %s",
                       GetID2Lang().at(lang_id).c_str());
    }

    return lang_id;
  }

  std::pair<Ort::Value, Ort::Value> GetInitialSelfKVCache() {
    std::array<int64_t, 4> shape{n_text_layer_, 1, n_text_ctx_, n_text_state_};

    Ort::Value n_layer_self_k_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    Ort::Value n_layer_self_v_cache = Ort::Value::CreateTensor<float>(
        Allocator(), shape.data(), shape.size());

    auto n = shape[0] * shape[1] * shape[2] * shape[3];

    float *p_k = n_layer_self_k_cache.GetTensorMutableData<float>();
    float *p_v = n_layer_self_v_cache.GetTensorMutableData<float>();

    memset(p_k, 0, sizeof(float) * n);
    memset(p_v, 0, sizeof(float) * n);

    return {std::move(n_layer_self_k_cache), std::move(n_layer_self_v_cache)};
  }

  OrtAllocator *Allocator() { return allocator_; }

  const std::vector<int64_t> &GetInitialTokens() const { return sot_sequence_; }

  const std::vector<int32_t> &GetAllLanguageIDs() const {
    return all_language_tokens_;
  }

  const std::unordered_map<std::string, int32_t> &GetLang2ID() const {
    return lang2id_;
  }

  const std::unordered_map<int32_t, std::string> &GetID2Lang() const {
    return id2lang_;
  }

  int32_t NoTimeStampsToken() const { return no_timestamps_; }

  int32_t EOT() const { return eot_; }

  int32_t SOT() const { return sot_; }

  int32_t TextCtx() const { return n_text_ctx_; }

  int32_t VocabSize() const { return n_vocab_; }

  int32_t FeatureDim() const { return n_mels_; }

  int32_t Translate() const { return translate_; }

  bool IsMultiLingual() const { return is_multilingual_; }

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
    SHERPA_ONNX_READ_META_DATA(n_mels_, "n_mels");
    SHERPA_ONNX_READ_META_DATA(n_text_layer_, "n_text_layer");
    SHERPA_ONNX_READ_META_DATA(n_text_ctx_, "n_text_ctx");
    SHERPA_ONNX_READ_META_DATA(n_text_state_, "n_text_state");
    SHERPA_ONNX_READ_META_DATA(n_vocab_, "n_vocab");
    SHERPA_ONNX_READ_META_DATA(sot_, "sot");
    SHERPA_ONNX_READ_META_DATA(eot_, "eot");
    SHERPA_ONNX_READ_META_DATA(blank_, "blank_id");
    SHERPA_ONNX_READ_META_DATA(translate_, "translate");
    SHERPA_ONNX_READ_META_DATA(transcribe_, "transcribe");
    SHERPA_ONNX_READ_META_DATA(is_multilingual_, "is_multilingual");
    SHERPA_ONNX_READ_META_DATA(no_timestamps_, "no_timestamps");
    SHERPA_ONNX_READ_META_DATA(no_speech_, "no_speech");
    SHERPA_ONNX_READ_META_DATA_VEC(sot_sequence_, "sot_sequence");

    if (is_multilingual_) {
      SHERPA_ONNX_READ_META_DATA_VEC(all_language_tokens_,
                                     "all_language_tokens");
      SHERPA_ONNX_READ_META_DATA_VEC_STRING(all_language_codes_,
                                            "all_language_codes");
      if (all_language_tokens_.size() != all_language_codes_.size()) {
        SHERPA_ONNX_LOGE("# lang_id: %d != # lang_code: %d",
                         static_cast<int32_t>(all_language_tokens_.size()),
                         static_cast<int32_t>(all_language_codes_.size()));
        exit(-1);
      }

      for (int32_t i = 0;
           i != static_cast<int32_t>(all_language_tokens_.size()); ++i) {
        lang2id_[all_language_codes_[i]] = all_language_tokens_[i];
        id2lang_[all_language_tokens_[i]] = all_language_codes_[i];
      }
    }
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
  }

  void InitCudaIOBinding() {
    use_cuda_iobinding_ = (!is_cpu_provider_ && IsCudaProvider(GetProvider()));
    if (use_cuda_iobinding_) {
      // Use device 0 by default. SessionOptions() in sherpa-onnx usually
      // configures the CUDA EP device; binding here only affects output memory.
      cuda_mem_info_ = std::make_unique<Ort::MemoryInfo>(
          "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
  }

  std::string GetProvider() const {
    if (!config_.provider.empty()) {
      return config_.provider;
    }
    return lid_config_.provider;
  }

  OfflineModelConfig config_;
  SpokenLanguageIdentificationConfig lid_config_;
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

  std::vector<int32_t> all_language_tokens_;
  std::vector<std::string> all_language_codes_;
  std::unordered_map<std::string, int32_t> lang2id_;
  std::unordered_map<int32_t, std::string> id2lang_;

  // model meta data
  int32_t n_mels_ = 80;
  int32_t n_text_layer_ = 0;
  int32_t n_text_ctx_ = 0;
  int32_t n_text_state_ = 0;
  int32_t n_vocab_ = 0;
  int32_t sot_ = 0;
  int32_t eot_ = 0;
  int32_t blank_ = 0;
  int32_t translate_ = 0;
  int32_t transcribe_ = 0;
  int32_t no_timestamps_ = 0;
  int32_t no_speech_ = 0;
  int32_t is_multilingual_ = 0;
  std::vector<int64_t> sot_sequence_;
};

OfflineWhisperModel::OfflineWhisperModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineWhisperModel::OfflineWhisperModel(
    const SpokenLanguageIdentificationConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineWhisperModel::OfflineWhisperModel(Manager *mgr,
                                         const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

template <typename Manager>
OfflineWhisperModel::OfflineWhisperModel(
    Manager *mgr, const SpokenLanguageIdentificationConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineWhisperModel::~OfflineWhisperModel() = default;

std::pair<Ort::Value, Ort::Value> OfflineWhisperModel::ForwardEncoder(
    Ort::Value features) const {
  return impl_->ForwardEncoder(std::move(features));
}

std::tuple<Ort::Value, Ort::Value, Ort::Value, Ort::Value, Ort::Value,
           Ort::Value>
OfflineWhisperModel::ForwardDecoder(Ort::Value tokens,
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

int32_t OfflineWhisperModel::DetectLanguage(Ort::Value &cross_k,    // NOLINT
                                            Ort::Value &cross_v) {  // NOLINT
  return impl_->DetectLanguage(cross_k, cross_v);
}

std::pair<Ort::Value, Ort::Value> OfflineWhisperModel::GetInitialSelfKVCache()
    const {
  return impl_->GetInitialSelfKVCache();
}

OrtAllocator *OfflineWhisperModel::Allocator() const {
  return impl_->Allocator();
}

const std::vector<int64_t> &OfflineWhisperModel::GetInitialTokens() const {
  return impl_->GetInitialTokens();
}

const std::vector<int32_t> &OfflineWhisperModel::GetAllLanguageIDs() const {
  return impl_->GetAllLanguageIDs();
}

const std::unordered_map<std::string, int32_t> &
OfflineWhisperModel::GetLang2ID() const {
  return impl_->GetLang2ID();
}

const std::unordered_map<int32_t, std::string> &
OfflineWhisperModel::GetID2Lang() const {
  return impl_->GetID2Lang();
}

int32_t OfflineWhisperModel::NoTimeStampsToken() const {
  return impl_->NoTimeStampsToken();
}

int32_t OfflineWhisperModel::EOT() const { return impl_->EOT(); }

int32_t OfflineWhisperModel::SOT() const { return impl_->SOT(); }

int32_t OfflineWhisperModel::TextCtx() const { return impl_->TextCtx(); }

int32_t OfflineWhisperModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineWhisperModel::FeatureDim() const { return impl_->FeatureDim(); }

int32_t OfflineWhisperModel::Translate() const { return impl_->Translate(); }

bool OfflineWhisperModel::IsMultiLingual() const {
  return impl_->IsMultiLingual();
}

void OfflineWhisperModel::NormalizeFeatures(float *features, int32_t num_frames,
                                            int32_t feat_dim) {
  NormalizeWhisperFeatures(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template OfflineWhisperModel::OfflineWhisperModel(
    AAssetManager *mgr, const OfflineModelConfig &config);

template OfflineWhisperModel::OfflineWhisperModel(
    AAssetManager *mgr, const SpokenLanguageIdentificationConfig &config);
#endif

#if __OHOS__
template OfflineWhisperModel::OfflineWhisperModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);

template OfflineWhisperModel::OfflineWhisperModel(
    NativeResourceManager *mgr,
    const SpokenLanguageIdentificationConfig &config);
#endif

}  // namespace sherpa_onnx
