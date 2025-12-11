// sherpa-onnx/csrc/axera/online-zipformer-transducer-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/online-zipformer-transducer-model-axera.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
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

#include "ax_engine_api.h"  // NOLINT
#include "ax_sys_api.h"     // NOLINT
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OnlineZipformerTransducerModelAxera::Impl {
 public:
  ~Impl() {
    FreeIO(&encoder_io_data_);
    FreeIO(&decoder_io_data_);
    FreeIO(&joiner_io_data_);

    if (encoder_handle_) {
      AX_ENGINE_DestroyHandle(encoder_handle_);
    }
    if (decoder_handle_) {
      AX_ENGINE_DestroyHandle(decoder_handle_);
    }
    if (joiner_handle_) {
      AX_ENGINE_DestroyHandle(joiner_handle_);
    }
  }

  explicit Impl(const OnlineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(config_.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config_.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config_.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(mgr, config_.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config_.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config_.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }
  }

  std::vector<std::vector<uint8_t>> GetEncoderInitStates() const {
    if (!encoder_io_info_ || encoder_io_info_->nInputSize <= 1) {
      return {};
    }

    uint32_t num_states = encoder_io_info_->nInputSize - 1;
    std::vector<std::vector<uint8_t>> states(num_states);

    for (uint32_t i = 1; i < encoder_io_info_->nInputSize; ++i) {
      const auto &in_meta = encoder_io_info_->pInputs[i];
      uint32_t bytes = in_meta.nSize;
      states[i - 1].resize(bytes);
      std::fill(states[i - 1].begin(), states[i - 1].end(), 0);
    }

    return states;
  }

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> RunEncoder(
      std::vector<float> features, std::vector<std::vector<uint8_t>> states) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!encoder_io_info_ || encoder_io_info_->nInputSize == 0) {
      SHERPA_ONNX_LOGE("Encoder io_info is not initialized");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &in0 = encoder_io_info_->pInputs[0];
    size_t bytes0 = in0.nSize;
    if (bytes0 != features.size() * sizeof(float)) {
      SHERPA_ONNX_LOGE(
          "Encoder feature size mismatch. model expects %u bytes, got %zu "
          "bytes",
          in0.nSize, features.size() * sizeof(float));
      SHERPA_ONNX_EXIT(-1);
    }
    std::memcpy(encoder_io_data_.pInputs[0].pVirAddr, features.data(), bytes0);

    uint32_t num_states = encoder_io_info_->nInputSize - 1;
    if (states.size() != num_states) {
      SHERPA_ONNX_LOGE("states.size() expected %u, but got %zu", num_states,
                       states.size());
      SHERPA_ONNX_EXIT(-1);
    }

    for (uint32_t i = 0; i < num_states; ++i) {
      const auto &in_meta = encoder_io_info_->pInputs[i + 1];
      size_t bytes = in_meta.nSize;
      if (bytes != states[i].size()) {
        SHERPA_ONNX_LOGE(
            "Encoder state %u size mismatch. model expects %u bytes, got %zu",
            i, in_meta.nSize, states[i].size());
        SHERPA_ONNX_EXIT(-1);
      }
      std::memcpy(encoder_io_data_.pInputs[i + 1].pVirAddr, states[i].data(),
                  bytes);
    }

    auto ret = AX_ENGINE_RunSync(encoder_handle_, &encoder_io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync(encoder) failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    if (encoder_io_info_->nOutputSize == 0) {
      SHERPA_ONNX_LOGE("Encoder has no output tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out0 = encoder_io_info_->pOutputs[0];
    auto &out0_buf = encoder_io_data_.pOutputs[0];
    size_t out0_elems = out0.nSize / sizeof(float);
    std::vector<float> encoder_out(out0_elems);
    std::memcpy(encoder_out.data(), out0_buf.pVirAddr, out0.nSize);

    std::vector<std::vector<uint8_t>> next_states(num_states);
    uint32_t num_state_out = encoder_io_info_->nOutputSize - 1;
    if (num_state_out != num_states) {
      SHERPA_ONNX_LOGE(
          "Encoder number of state outputs (%u) != number of state inputs (%u)",
          num_state_out, num_states);
      SHERPA_ONNX_EXIT(-1);
    }

    for (uint32_t i = 0; i < num_states; ++i) {
      const auto &out_meta = encoder_io_info_->pOutputs[i + 1];
      auto &out_buf = encoder_io_data_.pOutputs[i + 1];
      next_states[i].resize(out_meta.nSize);
      std::memcpy(next_states[i].data(), out_buf.pVirAddr, out_meta.nSize);
    }

    return {std::move(encoder_out), std::move(next_states)};
  }

  std::vector<float> RunDecoder(std::vector<int64_t> decoder_input) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!decoder_io_info_ || decoder_io_info_->nInputSize != 1) {
      SHERPA_ONNX_LOGE("Decoder expects exactly 1 input");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &in0 = decoder_io_info_->pInputs[0];
    size_t bytes = in0.nSize;
    if (bytes != decoder_input.size() * sizeof(int64_t)) {
      SHERPA_ONNX_LOGE(
          "Decoder input size mismatch. model expects %u bytes, got %zu bytes",
          in0.nSize, decoder_input.size() * sizeof(int64_t));
      SHERPA_ONNX_EXIT(-1);
    }

    std::memcpy(decoder_io_data_.pInputs[0].pVirAddr, decoder_input.data(),
                bytes);

    auto ret = AX_ENGINE_RunSync(decoder_handle_, &decoder_io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync(decoder) failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    if (decoder_io_info_->nOutputSize != 1) {
      SHERPA_ONNX_LOGE("Decoder expects exactly 1 output");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out0 = decoder_io_info_->pOutputs[0];
    auto &out_buf = decoder_io_data_.pOutputs[0];
    size_t out_elems = out0.nSize / sizeof(float);
    std::vector<float> decoder_out(out_elems);
    std::memcpy(decoder_out.data(), out_buf.pVirAddr, out0.nSize);

    return decoder_out;
  }

  std::vector<float> RunJoiner(const float *encoder_out,
                               const float *decoder_out) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!joiner_io_info_ || joiner_io_info_->nInputSize != 2) {
      SHERPA_ONNX_LOGE("Joiner expects exactly 2 inputs");
      SHERPA_ONNX_EXIT(-1);
    }

    {
      const auto &in0 = joiner_io_info_->pInputs[0];
      size_t bytes0 = in0.nSize;
      std::memcpy(joiner_io_data_.pInputs[0].pVirAddr, encoder_out, bytes0);
    }

    {
      const auto &in1 = joiner_io_info_->pInputs[1];
      size_t bytes1 = in1.nSize;
      std::memcpy(joiner_io_data_.pInputs[1].pVirAddr, decoder_out, bytes1);
    }

    auto ret = AX_ENGINE_RunSync(joiner_handle_, &joiner_io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync(joiner) failed, ret = %d", ret);
      SHERPA_ONNX_EXIT(-1);
    }

    if (joiner_io_info_->nOutputSize != 1) {
      SHERPA_ONNX_LOGE("Joiner expects exactly 1 output");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out0 = joiner_io_info_->pOutputs[0];
    auto &out_buf = joiner_io_data_.pOutputs[0];
    size_t out_elems = out0.nSize / sizeof(float);
    std::vector<float> joiner_out(out_elems);
    std::memcpy(joiner_out.data(), out_buf.pVirAddr, out0.nSize);

    return joiner_out;
  }

  int32_t ContextSize() const { return context_size_; }
  int32_t ChunkSize() const { return T_; }
  int32_t ChunkShift() const { return decode_chunk_len_; }
  int32_t VocabSize() const { return vocab_size_; }

  std::vector<int32_t> GetEncoderOutShape() const {
    if (!encoder_io_info_ || encoder_io_info_->nOutputSize == 0) {
      SHERPA_ONNX_LOGE("Encoder io_info is not initialized");
      SHERPA_ONNX_EXIT(-1);
    }

    const auto &out0 = encoder_io_info_->pOutputs[0];
    std::vector<int32_t> shape;
    shape.reserve(out0.nShapeSize);
    for (uint32_t i = 0; i < out0.nShapeSize; ++i) {
      shape.push_back(static_cast<int32_t>(out0.pShape[i]));
    }
    return shape;
  }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &encoder_handle_);
    InitInputOutputAttrs(encoder_handle_, config_.debug, &encoder_io_info_);
    PrepareIO(encoder_io_info_, &encoder_io_data_, config_.debug);

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
      SHERPA_ONNX_LOGE(
          "Please provide complete Zipformer meta in "
          "config.zipformer_meta for Axera model (no custom meta): "
          "encoder_dims, attention_dims, num_encoder_layers, "
          "cnn_module_kernels, left_context_len, T, decode_chunk_len, "
          "context_size.");
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      auto print = [](const std::vector<int32_t> &v, const char *name) {
        std::ostringstream os;
        os << name << ": ";
        for (auto i : v) {
          os << i << " ";
        }
#if __OHOS__
        SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
        SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
      };

      print(encoder_dims_, "encoder_dims");
      print(attention_dims_, "attention_dims");
      print(num_encoder_layers_, "num_encoder_layers");
      print(cnn_module_kernels_, "cnn_module_kernels");
      print(left_context_len_, "left_context_len");
#if __OHOS__
      SHERPA_ONNX_LOGE("T: %{public}d", T_);
      SHERPA_ONNX_LOGE("decode_chunk_len_: %{public}d", decode_chunk_len_);
#else
      SHERPA_ONNX_LOGE("T: %d", T_);
      SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);
#endif
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &decoder_handle_);
    InitInputOutputAttrs(decoder_handle_, config_.debug, &decoder_io_info_);
    PrepareIO(decoder_io_info_, &decoder_io_data_, config_.debug);

    if (!decoder_io_info_ || decoder_io_info_->nInputSize < 1) {
      SHERPA_ONNX_LOGE("Decoder has no input tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    auto &in0 = decoder_io_info_->pInputs[0];
    if (in0.nShapeSize < 2) {
      SHERPA_ONNX_LOGE("Decoder input rank is too small (nShapeSize = %u)",
                       in0.nShapeSize);
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t shape_context_size = in0.pShape[1];
    if (shape_context_size != context_size_) {
      SHERPA_ONNX_LOGE(
          "Decoder context_size mismatch. From config: %d, "
          "from decoder input shape: %d, use shape value.",
          context_size_, shape_context_size);
      context_size_ = shape_context_size;
    }

    if (config_.debug) {
      SHERPA_ONNX_LOGE("context_size: %d", context_size_);
    }
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &joiner_handle_);
    InitInputOutputAttrs(joiner_handle_, config_.debug, &joiner_io_info_);
    PrepareIO(joiner_io_info_, &joiner_io_data_, config_.debug);

    if (!joiner_io_info_ || joiner_io_info_->nOutputSize < 1) {
      SHERPA_ONNX_LOGE("Joiner has no output tensor");
      SHERPA_ONNX_EXIT(-1);
    }

    auto &out0 = joiner_io_info_->pOutputs[0];
    if (out0.nShapeSize < 2) {
      SHERPA_ONNX_LOGE("Joiner output rank is too small (nShapeSize = %u)",
                       out0.nShapeSize);
      SHERPA_ONNX_EXIT(-1);
    }

    vocab_size_ = out0.pShape[1];

    if (config_.debug) {
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
    }
  }

 private:
  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;
  OnlineModelConfig config_;

  AX_ENGINE_HANDLE encoder_handle_ = nullptr;
  AX_ENGINE_HANDLE decoder_handle_ = nullptr;
  AX_ENGINE_HANDLE joiner_handle_ = nullptr;

  AX_ENGINE_IO_INFO_T *encoder_io_info_ = nullptr;
  AX_ENGINE_IO_INFO_T *decoder_io_info_ = nullptr;
  AX_ENGINE_IO_INFO_T *joiner_io_info_ = nullptr;

  AX_ENGINE_IO_T encoder_io_data_;
  AX_ENGINE_IO_T decoder_io_data_;
  AX_ENGINE_IO_T joiner_io_data_;

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

OnlineZipformerTransducerModelAxera::~OnlineZipformerTransducerModelAxera() =
    default;

OnlineZipformerTransducerModelAxera::OnlineZipformerTransducerModelAxera(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineZipformerTransducerModelAxera::OnlineZipformerTransducerModelAxera(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<std::vector<uint8_t>>
OnlineZipformerTransducerModelAxera::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>>
OnlineZipformerTransducerModelAxera::RunEncoder(
    std::vector<float> features,
    std::vector<std::vector<uint8_t>> states) const {
  return impl_->RunEncoder(std::move(features), std::move(states));
}

std::vector<float> OnlineZipformerTransducerModelAxera::RunDecoder(
    std::vector<int64_t> decoder_input) const {
  return impl_->RunDecoder(std::move(decoder_input));
}

std::vector<float> OnlineZipformerTransducerModelAxera::RunJoiner(
    const float *encoder_out, const float *decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineZipformerTransducerModelAxera::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelAxera::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineZipformerTransducerModelAxera::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerTransducerModelAxera::VocabSize() const {
  return impl_->VocabSize();
}

std::vector<int32_t> OnlineZipformerTransducerModelAxera::GetEncoderOutShape()
    const {
  return impl_->GetEncoderOutShape();
}

#if __ANDROID_API__ >= 9
template OnlineZipformerTransducerModelAxera::
    OnlineZipformerTransducerModelAxera(AAssetManager *mgr,
                                        const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformerTransducerModelAxera::
    OnlineZipformerTransducerModelAxera(NativeResourceManager *mgr,
                                        const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx