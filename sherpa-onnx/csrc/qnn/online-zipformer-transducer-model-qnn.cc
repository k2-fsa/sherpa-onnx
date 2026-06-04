// sherpa-onnx/csrc/qnn/online-zipformer-transducer-model-qnn.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/qnn/online-zipformer-transducer-model-qnn.h"

#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <numeric>
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
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/qnn/macros.h"
#include "sherpa-onnx/csrc/qnn/qnn-backend.h"
#include "sherpa-onnx/csrc/qnn/qnn-model.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

static bool NeedsCacheTranspose(const std::string &name) {
  return name.rfind("cached_key_", 0) == 0 ||
         name.rfind("cached_val_", 0) == 0 ||
         name.rfind("cached_val2_", 0) == 0;
}

static std::vector<float> Transpose0123To0231(const float *x, int64_t d0,
                                              int64_t d1, int64_t d2,
                                              int64_t d3) {
  std::vector<float> y(static_cast<size_t>(d0) * d1 * d2 * d3);

  for (int64_t i0 = 0; i0 != d0; ++i0) {
    for (int64_t i1 = 0; i1 != d1; ++i1) {
      for (int64_t i2 = 0; i2 != d2; ++i2) {
        for (int64_t i3 = 0; i3 != d3; ++i3) {
          int64_t src = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
          int64_t dst = ((i0 * d2 + i2) * d3 + i3) * d1 + i1;
          y[dst] = x[src];
        }
      }
    }
  }

  return y;
}

static int64_t NumElements(const std::vector<int64_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

}  // namespace

class OnlineZipformerTransducerModelQnn::Impl {
 public:
  Impl(const OnlineModelConfig &config, int32_t feature_dim)
      : config_(config), expected_feature_dim_(feature_dim) {
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config, int32_t feature_dim)
      : config_(config), expected_feature_dim_(feature_dim) {
    SHERPA_ONNX_LOGE(
        "Please copy all files from assets to SD card and set assetManager to "
        "null");
    SHERPA_ONNX_EXIT(-1);
  }

  std::vector<OnlineStreamStateTensor> GetEncoderInitStates() const {
    std::vector<OnlineStreamStateTensor> ans;
    ans.reserve(state_storage_shapes_.size());
    for (size_t i = 0; i != state_storage_shapes_.size(); ++i) {
      OnlineStreamStateTensor state;
      int64_t n = NumElements(state_storage_shapes_[i]);
      if (state_is_int32_[i]) {
        state.int32_data.resize(n, 0);
      } else {
        state.float_data.resize(n, 0.f);
      }
      ans.push_back(std::move(state));
    }

    return ans;
  }

  std::vector<float> RunEncoder(std::vector<float> features, int32_t num_frames,
                               std::vector<OnlineStreamStateTensor> *states) const {
    if (!states) {
      SHERPA_ONNX_LOGE("states pointer is null");
      SHERPA_ONNX_EXIT(-1);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    features = Transpose(features.data(), num_frames, feat_dim_);

    encoder_->SetInputTensorData(encoder_input_name_, features.data(),
                                features.size());

    if (has_processed_lens_) {
      RunEncoderV2(states);
    } else {
      RunEncoderV1(states);
    }

    encoder_->Run();

    std::vector<float> encoder_out =
        encoder_->GetOutputTensorData("encoder_out");
    for (size_t i = 0; i != state_output_names_.size(); ++i) {
      const auto &name = state_output_names_[i];
      if (state_is_int32_[i]) {
        (*states)[i].int32_data = encoder_->GetOutputTensorDataInt32(name);
      } else {
        (*states)[i].float_data = encoder_->GetOutputTensorData(name);
      }
    }

    return encoder_out;
  }

  std::vector<float> RunDecoder(const std::vector<int32_t> &tokens) const {
    std::lock_guard<std::mutex> lock(mutex_);

    decoder_->SetInputTensorData(decoder_input_name_, tokens.data(),
                                 tokens.size());
    decoder_->Run();

    return decoder_->GetOutputTensorData(decoder_output_name_);
  }

  std::vector<float> RunJoiner(const float *encoder_out,
                               const std::vector<float> &decoder_out) const {
    std::lock_guard<std::mutex> lock(mutex_);

    joiner_->SetInputTensorData(joiner_encoder_input_name_, encoder_out,
                                encoder_out_dim_);
    joiner_->SetInputTensorData(joiner_decoder_input_name_, decoder_out.data(),
                                decoder_out_dim_);
    joiner_->Run();

    return joiner_->GetOutputTensorData(joiner_output_name_);
  }

  int32_t ContextSize() const { return context_size_; }
  int32_t ChunkSize() const { return chunk_size_; }
  int32_t ChunkShift() const { return chunk_shift_; }
  int32_t VocabSize() const { return vocab_size_; }
  int32_t FeatureDim() const { return feat_dim_; }
  int32_t EncoderDim() const { return encoder_out_dim_; }

 private:
  // Set encoder state inputs for V1 model (cached_* states, some need transpose)
  void RunEncoderV1(std::vector<OnlineStreamStateTensor> *states) const {
    for (size_t i = 0; i != state_input_names_.size(); ++i) {
      const auto &name = state_input_names_[i];
      if (!(*states)[i].int32_data.empty()) {
        const auto *p = (*states)[i].int32_data.data();
        int32_t n = static_cast<int32_t>((*states)[i].int32_data.size());
        encoder_->SetInputTensorData(name, p, n);
      } else {
        const auto *p = (*states)[i].float_data.data();
        int32_t n = static_cast<int32_t>((*states)[i].float_data.size());
        if (state_needs_transpose_[i]) {
          const auto &shape = state_storage_shapes_[i];
          auto transposed =
              Transpose0123To0231(p, shape[0], shape[1], shape[2], shape[3]);
          encoder_->SetInputTensorData(name, transposed.data(),
                                       static_cast<int32_t>(transposed.size()));
        } else {
          encoder_->SetInputTensorData(name, p, n);
        }
      }
    }
  }

  // Set encoder state inputs for V2 model (processed_lens, no transpose)
  void RunEncoderV2(std::vector<OnlineStreamStateTensor> *states) const {
    for (size_t i = 0; i != state_input_names_.size(); ++i) {
      const auto &name = state_input_names_[i];
      if (!(*states)[i].int32_data.empty()) {
        const auto *p = (*states)[i].int32_data.data();
        int32_t n = static_cast<int32_t>((*states)[i].int32_data.size());
        encoder_->SetInputTensorData(name, p, n);
      } else {
        const auto *p = (*states)[i].float_data.data();
        int32_t n = static_cast<int32_t>((*states)[i].float_data.size());
        encoder_->SetInputTensorData(name, p, n);
      }
    }
  }

  void Init() {
   ParseContextBinaries();
   encoder_backend_ = std::make_unique<QnnBackend>(
       config_.transducer.qnn_config.backend_lib, config_.debug);
   decoder_backend_ = std::make_unique<QnnBackend>(
       config_.transducer.qnn_config.backend_lib, config_.debug);
   joiner_backend_ = std::make_unique<QnnBackend>(
       config_.transducer.qnn_config.backend_lib, config_.debug);

    InitEncoder();
    InitDecoder();
    InitJoiner();

    CheckEncoder();
    CheckDecoder();
    CheckJoiner();
  }

  void ParseContextBinaries() {
    SplitStringToVector(config_.transducer.qnn_config.context_binary, ",", true,
                        &context_binaries_);
    if (!context_binaries_.empty() && context_binaries_.size() != 3) {
      SHERPA_ONNX_LOGE(
          "There should be 3 files for online transducer context binary. "
          "Actual: %d. '%s'",
          static_cast<int32_t>(context_binaries_.size()),
          config_.transducer.qnn_config.context_binary.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void CreateContextBinary(QnnModel *model,
                           const std::string &context_binary) const {
    if (config_.debug) {
      SHERPA_ONNX_LOGE("Creating context binary '%s'.", context_binary.c_str());
    }

    bool ok = model->SaveBinaryContext(context_binary);

    if (!ok) {
      SHERPA_ONNX_LOGE("Failed to save context binary to '%s'",
                       context_binary.c_str());
    }

    if (config_.debug && ok) {
      SHERPA_ONNX_LOGE("Saved context binary to '%s'.", context_binary.c_str());
      SHERPA_ONNX_LOGE("Remember to also provide libQnnSystem.so.");
    }
  }

  void InitModelFromLib(const std::string &filename, QnnBackend *backend,
                        std::unique_ptr<QnnModel> *model) const {
    backend->InitContext();
    *model = std::make_unique<QnnModel>(filename, backend, config_.debug);
  }

  void InitModelFromContextBinary(const std::string &filename,
                                  QnnBackend *backend,
                                  std::unique_ptr<QnnModel> *model) const {
    if (config_.transducer.qnn_config.system_lib.empty()) {
      SHERPA_ONNX_LOGE(
          "You should provide --transducer.qnn-system-lib if you also provide "
          "context binary");
      SHERPA_ONNX_EXIT(-1);
    }

    *model = std::make_unique<QnnModel>(
        filename, config_.transducer.qnn_config.system_lib, backend,
        BinaryContextTag{}, config_.debug);
  }

  void InitComponent(const std::string &lib_filename,
                     const std::string &context_binary, const char *name,
                     QnnBackend *backend,
                     std::unique_ptr<QnnModel> *model) const {
    if (context_binary.empty()) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init from %s model lib '%s' since context binary is not given.",
            name, lib_filename.c_str());
      }

      InitModelFromLib(lib_filename, backend, model);
      return;
    }

    if (!FileExists(context_binary)) {
      if (config_.debug) {
        SHERPA_ONNX_LOGE(
            "Init %s from model lib '%s' since context binary '%s' does not "
            "exist",
            name, lib_filename.c_str(), context_binary.c_str());
      }

      InitModelFromLib(lib_filename, backend, model);
      CreateContextBinary(model->get(), context_binary);
    } else {
      if (config_.debug) {
        SHERPA_ONNX_LOGE("Init from %s context binary '%s'", name,
                         context_binary.c_str());
      }

      InitModelFromContextBinary(context_binary, backend, model);
    }
  }

  void InitEncoder() {
    InitComponent(config_.transducer.encoder,
                  context_binaries_.empty() ? "" : context_binaries_[0],
                  "encoder", encoder_backend_.get(), &encoder_);
  }

  void InitDecoder() {
    InitComponent(config_.transducer.decoder,
                  context_binaries_.empty() ? "" : context_binaries_[1],
                  "decoder", decoder_backend_.get(), &decoder_);
  }

  void InitJoiner() {
    InitComponent(config_.transducer.joiner,
                  context_binaries_.empty() ? "" : context_binaries_[2],
                  "joiner", joiner_backend_.get(), &joiner_);
  }

  void CheckEncoder() {
    if (!encoder_->HasTensor("encoder_out")) {
      SHERPA_ONNX_LOGE("The encoder model does not have output 'encoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }

    has_processed_lens_ = encoder_->HasTensor("processed_lens");

    encoder_input_name_.clear();
    for (const auto &name : encoder_->InputTensorNames()) {
      if (name == "x") {
        encoder_input_name_ = name;
      } else if (has_processed_lens_) {
        // V2: all non-"x" inputs are streaming states
        state_input_names_.push_back(name);
      } else if (name.rfind("cached_", 0) == 0) {
        // V1: only "cached_*" inputs are streaming states
        state_input_names_.push_back(name);
      }
    }

    if (encoder_input_name_.empty()) {
      SHERPA_ONNX_LOGE("The encoder model does not have input 'x'");
      SHERPA_ONNX_EXIT(-1);
    }

    if (state_input_names_.empty()) {
      SHERPA_ONNX_LOGE("The encoder model does not contain streaming states");
      SHERPA_ONNX_EXIT(-1);
    }

    auto x_shape = encoder_->TensorShape(encoder_input_name_);
    if (x_shape.size() != 3 || x_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The encoder input x should be of shape [1, ?, ?]");
      SHERPA_ONNX_EXIT(-1);
    }

    if (x_shape[1] != expected_feature_dim_) {
      SHERPA_ONNX_LOGE(
          "The encoder input x should be of shape [1, %d, T]. Given [1, %d, %d]",
          expected_feature_dim_, static_cast<int32_t>(x_shape[1]),
          static_cast<int32_t>(x_shape[2]));
      SHERPA_ONNX_EXIT(-1);
    }
    feat_dim_ = x_shape[1];
    chunk_size_ = x_shape[2];

    auto out_shape = encoder_->TensorShape("encoder_out");
    if (out_shape.size() != 3 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The encoder output should be of shape [1, ?, ?]");
      SHERPA_ONNX_EXIT(-1);
    }

    encoder_out_dim_ = out_shape[2];
    chunk_shift_ = out_shape[1] * 4;

    for (const auto &name : state_input_names_) {
      std::string new_name = "new_" + name;
      if (!encoder_->HasTensor(new_name)) {
        SHERPA_ONNX_LOGE("The encoder model does not have output '%s'",
                         new_name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }

      state_output_names_.push_back(new_name);

      auto output_shape = encoder_->TensorShape(new_name);
      state_storage_shapes_.emplace_back(output_shape.begin(),
                                         output_shape.end());

      if (has_processed_lens_) {
        state_is_int32_.push_back(name == "processed_lens");
        state_needs_transpose_.push_back(false);
      } else {
        state_is_int32_.push_back(name.rfind("cached_len_", 0) == 0);
        state_needs_transpose_.push_back(NeedsCacheTranspose(name));
      }
    }
  }

  void CheckDecoder() {
    const auto &input_names = decoder_->InputTensorNames();
    if (input_names.size() != 1) {
      SHERPA_ONNX_LOGE("Expected one decoder input. Given %d",
                       static_cast<int32_t>(input_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    decoder_input_name_ = input_names[0];
    auto in_shape = decoder_->TensorShape(decoder_input_name_);
    if (in_shape.size() != 2 || in_shape[0] != 1) {
      SHERPA_ONNX_LOGE("Expected decoder input shape [1, context_size]");
      SHERPA_ONNX_EXIT(-1);
    }
    context_size_ = in_shape[1];

    if (!decoder_->HasTensor("decoder_out")) {
      SHERPA_ONNX_LOGE("The decoder model does not have output 'decoder_out'");
      SHERPA_ONNX_EXIT(-1);
    }

    decoder_output_name_ = "decoder_out";
    auto out_shape = decoder_->TensorShape(decoder_output_name_);
    if (out_shape.size() != 2 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE(
          "The decoder output should be of shape [1, decoder_dim]");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &output_names = decoder_->OutputTensorNames();
    if (output_names.size() != 1 || output_names[0] != decoder_output_name_) {
      SHERPA_ONNX_LOGE(
          "Expected one decoder output named 'decoder_out'. Given %d outputs",
          static_cast<int32_t>(output_names.size()));
      SHERPA_ONNX_EXIT(-1);
    }
    decoder_out_dim_ = out_shape[1];
  }

  void CheckJoiner() {
    if (!joiner_->HasTensor("logit")) {
      SHERPA_ONNX_LOGE("The joiner model does not have output 'logit'");
      SHERPA_ONNX_EXIT(-1);
    }

    joiner_encoder_input_name_ = "encoder_out";
    joiner_decoder_input_name_ = "decoder_out";

    if (!joiner_->HasTensor(joiner_encoder_input_name_)) {
      SHERPA_ONNX_LOGE("The joiner model does not have input '%s'",
                       joiner_encoder_input_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!joiner_->HasTensor(joiner_decoder_input_name_)) {
      SHERPA_ONNX_LOGE("The joiner model does not have input '%s'",
                       joiner_decoder_input_name_.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    auto encoder_in_shape = joiner_->TensorShape(joiner_encoder_input_name_);
    if (encoder_in_shape.size() != 2 || encoder_in_shape[0] != 1 ||
        encoder_in_shape[1] != encoder_out_dim_) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be of shape [1, %d]",
          joiner_encoder_input_name_.c_str(), encoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto decoder_in_shape = joiner_->TensorShape(joiner_decoder_input_name_);
    if (decoder_in_shape.size() != 2 || decoder_in_shape[0] != 1 ||
        decoder_in_shape[1] != decoder_out_dim_) {
      SHERPA_ONNX_LOGE(
          "The joiner input '%s' should be of shape [1, %d]",
          joiner_decoder_input_name_.c_str(), decoder_out_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    auto out_shape = joiner_->TensorShape("logit");
    if (out_shape.size() != 2 || out_shape[0] != 1) {
      SHERPA_ONNX_LOGE("The joiner output should be of shape [1, vocab_size]");
      SHERPA_ONNX_EXIT(-1);
    }

    joiner_output_name_ = "logit";
    vocab_size_ = out_shape[1];
  }

 private:
 mutable std::mutex mutex_;

 OnlineModelConfig config_;
 int32_t expected_feature_dim_ = 80;

  std::unique_ptr<QnnBackend> encoder_backend_;
  std::unique_ptr<QnnBackend> decoder_backend_;
  std::unique_ptr<QnnBackend> joiner_backend_;

  std::unique_ptr<QnnModel> encoder_;
  std::unique_ptr<QnnModel> decoder_;
  std::unique_ptr<QnnModel> joiner_;
  std::vector<std::string> context_binaries_;

  std::string encoder_input_name_;
  std::vector<std::string> state_input_names_;
  std::vector<std::string> state_output_names_;

  std::string decoder_input_name_;
  std::string decoder_output_name_;

  std::string joiner_encoder_input_name_;
  std::string joiner_decoder_input_name_;
  std::string joiner_output_name_;

  std::vector<std::vector<int64_t>> state_storage_shapes_;
  std::vector<bool> state_is_int32_;
  std::vector<bool> state_needs_transpose_;

  bool has_processed_lens_ = false;

  int32_t chunk_size_ = 0;
  int32_t chunk_shift_ = 0;
  int32_t feat_dim_ = 0;
  int32_t encoder_out_dim_ = 0;
  int32_t decoder_out_dim_ = 0;
  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
};

OnlineZipformerTransducerModelQnn::OnlineZipformerTransducerModelQnn(
    const OnlineModelConfig &config, int32_t feature_dim)
    : impl_(std::make_unique<Impl>(config, feature_dim)) {}

template <typename Manager>
OnlineZipformerTransducerModelQnn::OnlineZipformerTransducerModelQnn(
    Manager *mgr, const OnlineModelConfig &config, int32_t feature_dim)
    : impl_(std::make_unique<Impl>(mgr, config, feature_dim)) {}

OnlineZipformerTransducerModelQnn::~OnlineZipformerTransducerModelQnn() =
    default;

std::vector<OnlineStreamStateTensor>
OnlineZipformerTransducerModelQnn::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::vector<float> OnlineZipformerTransducerModelQnn::RunEncoder(
    std::vector<float> features, int32_t num_frames,
    std::vector<OnlineStreamStateTensor> *states) const {
  return impl_->RunEncoder(std::move(features), num_frames, states);
}

std::vector<float> OnlineZipformerTransducerModelQnn::RunDecoder(
    const std::vector<int32_t> &tokens) const {
  return impl_->RunDecoder(tokens);
}

std::vector<float> OnlineZipformerTransducerModelQnn::RunJoiner(
    const float *encoder_out, const std::vector<float> &decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineZipformerTransducerModelQnn::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelQnn::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineZipformerTransducerModelQnn::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerTransducerModelQnn::VocabSize() const {
  return impl_->VocabSize();
}

int32_t OnlineZipformerTransducerModelQnn::FeatureDim() const {
  return impl_->FeatureDim();
}

int32_t OnlineZipformerTransducerModelQnn::EncoderDim() const {
  return impl_->EncoderDim();
}

#if __ANDROID_API__ >= 9
template OnlineZipformerTransducerModelQnn::OnlineZipformerTransducerModelQnn(
    AAssetManager *mgr, const OnlineModelConfig &config, int32_t feature_dim);
#endif

#if __OHOS__
template OnlineZipformerTransducerModelQnn::OnlineZipformerTransducerModelQnn(
    NativeResourceManager *mgr, const OnlineModelConfig &config,
    int32_t feature_dim);
#endif

}  // namespace sherpa_onnx
