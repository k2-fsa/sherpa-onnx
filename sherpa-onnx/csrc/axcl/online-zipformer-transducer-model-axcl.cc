// sherpa-onnx/csrc/axera/online-zipformer-transducer-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/online-zipformer-transducer-model-axcl.h"

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/axcl/axcl-model.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

class OnlineZipformerTransducerModelAxcl::Impl {
 public:
  explicit Impl(const OnlineModelConfig &config) : config_(config) {
    encoder_ = std::make_unique<AxclModel>(config_.transducer.encoder);
    decoder_ = std::make_unique<AxclModel>(config_.transducer.decoder);
    joiner_ = std::make_unique<AxclModel>(config_.transducer.joiner);
    PostInit();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(mgr, config_.transducer.encoder);
      encoder_ = std::make_unique<AxclModel>(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(mgr, config_.transducer.decoder);
      decoder_ = std::make_unique<AxclModel>(buf.data(), buf.size());
    }
    {
      auto buf = ReadFile(mgr, config_.transducer.joiner);
      joiner_ = std::make_unique<AxclModel>(buf.data(), buf.size());
    }
    PostInit();
  }

  std::vector<std::vector<uint8_t>> GetEncoderInitStates() const {
    // encoder inputs: [0]=x, [1..]=states
    const auto &names = encoder_->InputTensorNames();
    if (names.empty()) return {};

    if (names[0] != "x") {
      SHERPA_ONNX_LOGE("Expected encoder input[0] name 'x', got '%s'",
                       names[0].c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<std::vector<uint8_t>> states;
    states.reserve(names.size() - 1);

    for (size_t i = 1; i < names.size(); ++i) {
      int32_t nbytes = encoder_->TensorSizeInBytes(names[i]);
      states.emplace_back(static_cast<size_t>(nbytes), 0);
    }
    return states;
  }

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> RunEncoder(
      std::vector<float> features, std::vector<std::vector<uint8_t>> states) {
    CheckSizeOrDie(*encoder_, "x",
                   static_cast<int32_t>(features.size() * sizeof(float)));
    if (!encoder_->SetInputTensorData("x", features.data(),
                                      static_cast<int32_t>(features.size()))) {
      SHERPA_ONNX_LOGE("Failed to set encoder input 'x'");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &in_names = encoder_->InputTensorNames();
    if (states.size() != in_names.size() - 1) {
      SHERPA_ONNX_LOGE("states.size() expected %zu, got %zu",
                       in_names.size() - 1, states.size());
      SHERPA_ONNX_EXIT(-1);
    }

    for (size_t i = 1; i < in_names.size(); ++i) {
      const std::string &name = in_names[i];
      int32_t expect = encoder_->TensorSizeInBytes(name);
      int32_t got = static_cast<int32_t>(states[i - 1].size());
      if (expect != got) {
        SHERPA_ONNX_LOGE("Encoder state '%s' size mismatch. expect %d, got %d",
                         name.c_str(), expect, got);
        SHERPA_ONNX_EXIT(-1);
      }

      if (StartsWith(name, "cached_len_")) {
        if (got % static_cast<int32_t>(sizeof(int32_t)) != 0) {
          SHERPA_ONNX_LOGE("cached_len tensor bytes not multiple of int32: %s",
                           name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
        const int32_t *p =
            reinterpret_cast<const int32_t *>(states[i - 1].data());
        int32_t n = got / sizeof(int32_t);
        if (!encoder_->SetInputTensorData(name, p, n)) {
          SHERPA_ONNX_LOGE("Failed to set int32 state: %s", name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
      } else {
        if (!encoder_->SetInputTensorDataRaw(name, states[i - 1].data(), got)) {
          SHERPA_ONNX_LOGE("Failed to set raw state: %s", name.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
      }
    }

    if (!encoder_->Run()) {
      SHERPA_ONNX_LOGE("encoder_->Run() failed");
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<float> encoder_out =
        encoder_->GetOutputTensorData("encoder_out");
    if (encoder_out.empty()) {
      SHERPA_ONNX_LOGE("Failed to get encoder_out");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out_names = encoder_->OutputTensorNames();
    if (out_names.empty() || out_names[0] != "encoder_out") {
      SHERPA_ONNX_LOGE("Expected encoder output[0] 'encoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<std::vector<uint8_t>> next_states;
    next_states.reserve(out_names.size() - 1);
    for (size_t i = 1; i < out_names.size(); ++i) {
      next_states.emplace_back(encoder_->GetOutputTensorDataRaw(out_names[i]));
      if (next_states.back().empty()) {
        SHERPA_ONNX_LOGE("Failed to get encoder output raw: %s",
                         out_names[i].c_str());
        SHERPA_ONNX_EXIT(-1);
      }
    }

    return {std::move(encoder_out), std::move(next_states)};
  }

  std::vector<float> RunDecoder(std::vector<int32_t> decoder_input) {
    CheckSizeOrDie(
        *decoder_, "y",
        static_cast<int32_t>(decoder_input.size() * sizeof(int32_t)));
    if (!decoder_->SetInputTensorData(
            "y", decoder_input.data(),
            static_cast<int32_t>(decoder_input.size()))) {
      SHERPA_ONNX_LOGE("Failed to set decoder input 'y'");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_->Run()) {
      SHERPA_ONNX_LOGE("decoder_->Run() failed");
      SHERPA_ONNX_EXIT(-1);
    }
    auto out = decoder_->GetOutputTensorData("decoder_out");
    if (out.empty()) {
      SHERPA_ONNX_LOGE("Failed to get decoder_out");
      SHERPA_ONNX_EXIT(-1);
    }
    return out;
  }

  std::vector<float> RunJoiner(const float *encoder_out,
                               const float *decoder_out) {
    int32_t enc_bytes = joiner_->TensorSizeInBytes("encoder_out");
    int32_t dec_bytes = joiner_->TensorSizeInBytes("decoder_out");

    if (enc_bytes % static_cast<int32_t>(sizeof(float)) != 0 ||
        dec_bytes % static_cast<int32_t>(sizeof(float)) != 0) {
      SHERPA_ONNX_LOGE("Joiner input bytes not multiple of sizeof(float)");
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t enc_n = enc_bytes / sizeof(float);
    int32_t dec_n = dec_bytes / sizeof(float);

    if (!joiner_->SetInputTensorData("encoder_out", encoder_out, enc_n)) {
      SHERPA_ONNX_LOGE("Failed to set joiner input encoder_out");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!joiner_->SetInputTensorData("decoder_out", decoder_out, dec_n)) {
      SHERPA_ONNX_LOGE("Failed to set joiner input decoder_out");
      SHERPA_ONNX_EXIT(-1);
    }

    if (!joiner_->Run()) {
      SHERPA_ONNX_LOGE("joiner_->Run() failed");
      SHERPA_ONNX_EXIT(-1);
    }

    auto out = joiner_->GetOutputTensorData("logit");
    if (out.empty()) {
      SHERPA_ONNX_LOGE("Failed to get joiner output logit");
      SHERPA_ONNX_EXIT(-1);
    }
    return out;
  }

  int32_t ContextSize() const { return context_size_; }
  int32_t ChunkSize() const { return T_; }
  int32_t ChunkShift() const { return decode_chunk_len_; }
  int32_t VocabSize() const { return vocab_size_; }

  std::vector<int32_t> GetEncoderOutShape() const {
    return encoder_->TensorShape("encoder_out");
  }

 private:
  static bool StartsWith(const std::string &s, const std::string &prefix) {
    return s.size() >= prefix.size() &&
           std::memcmp(s.data(), prefix.data(), prefix.size()) == 0;
  }

  static void CheckSizeOrDie(const AxclModel &m, const std::string &name,
                             int32_t expect_bytes) {
    if (!m.HasTensor(name)) {
      SHERPA_ONNX_LOGE("Missing tensor: %s", name.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    int32_t got = m.TensorSizeInBytes(name);
    if (got != expect_bytes) {
      SHERPA_ONNX_LOGE("Tensor %s bytes mismatch. expect %d, got %d",
                       name.c_str(), expect_bytes, got);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void PostInit() {
    if (!encoder_->IsInitialized() || !decoder_->IsInitialized() ||
        !joiner_->IsInitialized()) {
      SHERPA_ONNX_LOGE("Failed to initialize one of axcl models");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &m = config_.zipformer_meta;
    encoder_dims_ = m.encoder_dims;
    attention_dims_ = m.attention_dims;
    num_encoder_layers_ = m.num_encoder_layers;
    cnn_module_kernels_ = m.cnn_module_kernels;
    left_context_len_ = m.left_context_len;
    T_ = m.T;
    decode_chunk_len_ = m.decode_chunk_len;
    context_size_ = m.context_size;

    if (encoder_dims_.empty() || attention_dims_.empty() ||
        num_encoder_layers_.empty() || cnn_module_kernels_.empty() ||
        left_context_len_.empty() || T_ <= 0 || decode_chunk_len_ <= 0 ||
        context_size_ <= 0) {
      SHERPA_ONNX_LOGE("Incomplete zipformer_meta in config");
      SHERPA_ONNX_EXIT(-1);
    }

    auto s = joiner_->TensorShape("logit");
    if (s.size() < 2) {
      SHERPA_ONNX_LOGE("Joiner output 'logit' rank too small");
      SHERPA_ONNX_EXIT(-1);
    }
    vocab_size_ = s[1];

    if (!encoder_->HasTensor("x") || !encoder_->HasTensor("encoder_out")) {
      SHERPA_ONNX_LOGE("Encoder missing required tensors x/encoder_out");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!decoder_->HasTensor("y") || !decoder_->HasTensor("decoder_out")) {
      SHERPA_ONNX_LOGE("Decoder missing required tensors y/decoder_out");
      SHERPA_ONNX_EXIT(-1);
    }
    if (!joiner_->HasTensor("encoder_out") ||
        !joiner_->HasTensor("decoder_out") || !joiner_->HasTensor("logit")) {
      SHERPA_ONNX_LOGE("Joiner missing required tensors");
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  OnlineModelConfig config_;
  std::unique_ptr<AxclModel> encoder_;
  std::unique_ptr<AxclModel> decoder_;
  std::unique_ptr<AxclModel> joiner_;

  std::vector<int32_t> encoder_dims_;
  std::vector<int32_t> attention_dims_;
  std::vector<int32_t> num_encoder_layers_;
  std::vector<int32_t> cnn_module_kernels_;
  std::vector<int32_t> left_context_len_;

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;
  int32_t context_size_ = 2;
  int32_t vocab_size_ = 0;
};

OnlineZipformerTransducerModelAxcl::~OnlineZipformerTransducerModelAxcl() =
    default;

OnlineZipformerTransducerModelAxcl::OnlineZipformerTransducerModelAxcl(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineZipformerTransducerModelAxcl::OnlineZipformerTransducerModelAxcl(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<std::vector<uint8_t>>
OnlineZipformerTransducerModelAxcl::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>>
OnlineZipformerTransducerModelAxcl::RunEncoder(
    std::vector<float> features,
    std::vector<std::vector<uint8_t>> states) const {
  return impl_->RunEncoder(std::move(features), std::move(states));
}

std::vector<float> OnlineZipformerTransducerModelAxcl::RunDecoder(
    std::vector<int32_t> decoder_input) const {
  return impl_->RunDecoder(std::move(decoder_input));
}

std::vector<float> OnlineZipformerTransducerModelAxcl::RunJoiner(
    const float *encoder_out, const float *decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineZipformerTransducerModelAxcl::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelAxcl::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineZipformerTransducerModelAxcl::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerTransducerModelAxcl::VocabSize() const {
  return impl_->VocabSize();
}

std::vector<int32_t> OnlineZipformerTransducerModelAxcl::GetEncoderOutShape()
    const {
  return impl_->GetEncoderOutShape();
}

#if __ANDROID_API__ >= 9
template OnlineZipformerTransducerModelAxcl::OnlineZipformerTransducerModelAxcl(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformerTransducerModelAxcl::OnlineZipformerTransducerModelAxcl(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx