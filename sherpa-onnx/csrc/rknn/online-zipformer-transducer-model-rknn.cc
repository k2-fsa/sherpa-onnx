// sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"

#include <memory>
#include <sstream>
#include <string>
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
#include "sherpa-onnx/csrc/rknn/macros.h"
#include "sherpa-onnx/csrc/rknn/utils.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OnlineZipformerTransducerModelRknn::Impl {
 public:
  ~Impl() {
    auto ret = rknn_destroy(encoder_ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the encoder context");
    }

    ret = rknn_destroy(decoder_ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the decoder context");
    }

    ret = rknn_destroy(joiner_ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the joiner context");
    }
  }

  explicit Impl(const OnlineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }

    SetCoreMask(encoder_ctx_, config_.num_threads);
    SetCoreMask(decoder_ctx_, config_.num_threads);
    SetCoreMask(joiner_ctx_, config_.num_threads);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OnlineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(mgr, config.transducer.encoder);
      InitEncoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.decoder);
      InitDecoder(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.transducer.joiner);
      InitJoiner(buf.data(), buf.size());
    }

    SetCoreMask(encoder_ctx_, config_.num_threads);
    SetCoreMask(decoder_ctx_, config_.num_threads);
    SetCoreMask(joiner_ctx_, config_.num_threads);
  }

  // TODO(fangjun): Support Android

  std::vector<std::vector<uint8_t>> GetEncoderInitStates() const {
    // encoder_input_attrs_[0] is for the feature
    // encoder_input_attrs_[1:] is for states
    // so we use -1 here
    std::vector<std::vector<uint8_t>> states(encoder_input_attrs_.size() - 1);

    int32_t i = -1;
    for (auto &attr : encoder_input_attrs_) {
      i += 1;
      if (i == 0) {
        // skip processing the attr for features.
        continue;
      }

      if (attr.type == RKNN_TENSOR_FLOAT16) {
        states[i - 1].resize(attr.n_elems * sizeof(float));
      } else if (attr.type == RKNN_TENSOR_INT64) {
        states[i - 1].resize(attr.n_elems * sizeof(int64_t));
      } else {
        SHERPA_ONNX_LOGE("Unsupported tensor type: %d, %s", attr.type,
                         get_type_string(attr.type));
        SHERPA_ONNX_EXIT(-1);
      }
    }

    return states;
  }

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> RunEncoder(
      std::vector<float> features, std::vector<std::vector<uint8_t>> states) {
    std::vector<rknn_input> inputs(encoder_input_attrs_.size());

    for (int32_t i = 0; i < static_cast<int32_t>(inputs.size()); ++i) {
      auto &input = inputs[i];
      auto &attr = encoder_input_attrs_[i];
      input.index = attr.index;

      if (attr.type == RKNN_TENSOR_FLOAT16) {
        input.type = RKNN_TENSOR_FLOAT32;
      } else if (attr.type == RKNN_TENSOR_INT64) {
        input.type = RKNN_TENSOR_INT64;
      } else {
        SHERPA_ONNX_LOGE("Unsupported tensor type %d, %s", attr.type,
                         get_type_string(attr.type));
        SHERPA_ONNX_EXIT(-1);
      }

      input.fmt = attr.fmt;
      if (i == 0) {
        input.buf = reinterpret_cast<void *>(features.data());
        input.size = features.size() * sizeof(float);
      } else {
        input.buf = reinterpret_cast<void *>(states[i - 1].data());
        input.size = states[i - 1].size();
      }
    }

    std::vector<float> encoder_out(encoder_output_attrs_[0].n_elems);

    // Note(fangjun): We can reuse the memory from input argument `states`
    // auto next_states = GetEncoderInitStates();
    auto &next_states = states;

    std::vector<rknn_output> outputs(encoder_output_attrs_.size());
    for (int32_t i = 0; i < outputs.size(); ++i) {
      auto &output = outputs[i];
      auto &attr = encoder_output_attrs_[i];
      output.index = attr.index;
      output.is_prealloc = 1;

      if (attr.type == RKNN_TENSOR_FLOAT16) {
        output.want_float = 1;
      } else if (attr.type == RKNN_TENSOR_INT64) {
        output.want_float = 0;
      } else {
        SHERPA_ONNX_LOGE("Unsupported tensor type %d, %s", attr.type,
                         get_type_string(attr.type));
        SHERPA_ONNX_EXIT(-1);
      }

      if (i == 0) {
        output.size = encoder_out.size() * sizeof(float);
        output.buf = reinterpret_cast<void *>(encoder_out.data());
      } else {
        output.size = next_states[i - 1].size();
        output.buf = reinterpret_cast<void *>(next_states[i - 1].data());
      }
    }

    rknn_context encoder_ctx = 0;

    // https://github.com/rockchip-linux/rknpu2/blob/master/runtime/RK3588/Linux/librknn_api/include/rknn_api.h#L444C1-L444C75
    // rknn_dup_context(rknn_context* context_in, rknn_context* context_out);
    auto ret = rknn_dup_context(&encoder_ctx_, &encoder_ctx);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to duplicate the encoder ctx");

    ret = rknn_inputs_set(encoder_ctx, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set encoder inputs");

    ret = rknn_run(encoder_ctx, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run encoder");

    ret =
        rknn_outputs_get(encoder_ctx, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get encoder output");

    for (int32_t i = 0; i < next_states.size(); ++i) {
      const auto &attr = encoder_input_attrs_[i + 1];
      if (attr.n_dims == 4) {
        // TODO(fangjun): The ConvertNCHWtoNHWC is copied from
        // https://github.com/airockchip/rknn_model_zoo/blob/main/examples/zipformer/cpp/process.cc#L22
        // I don't understand why we need to do that.
        std::vector<uint8_t> dst(next_states[i].size());
        int32_t n = attr.dims[0];
        int32_t h = attr.dims[1];
        int32_t w = attr.dims[2];
        int32_t c = attr.dims[3];
        ConvertNCHWtoNHWC(
            reinterpret_cast<const float *>(next_states[i].data()), n, c, h, w,
            reinterpret_cast<float *>(dst.data()));
        next_states[i] = std::move(dst);
      }
    }

    rknn_destroy(encoder_ctx);

    return {std::move(encoder_out), std::move(next_states)};
  }

  std::vector<float> RunDecoder(std::vector<int64_t> decoder_input) {
    auto &attr = decoder_input_attrs_[0];
    rknn_input input;

    input.index = 0;
    input.type = RKNN_TENSOR_INT64;
    input.fmt = attr.fmt;
    input.buf = decoder_input.data();
    input.size = decoder_input.size() * sizeof(int64_t);

    std::vector<float> decoder_out(decoder_output_attrs_[0].n_elems);
    rknn_output output;
    output.index = decoder_output_attrs_[0].index;
    output.is_prealloc = 1;
    output.want_float = 1;
    output.size = decoder_out.size() * sizeof(float);
    output.buf = decoder_out.data();

    rknn_context decoder_ctx = 0;
    auto ret = rknn_dup_context(&decoder_ctx_, &decoder_ctx);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to duplicate the decoder ctx");

    ret = rknn_inputs_set(decoder_ctx, 1, &input);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set decoder inputs");

    ret = rknn_run(decoder_ctx, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run decoder");

    ret = rknn_outputs_get(decoder_ctx, 1, &output, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get decoder output");

    rknn_destroy(decoder_ctx);

    return decoder_out;
  }

  std::vector<float> RunJoiner(const float *encoder_out,
                               const float *decoder_out) {
    std::vector<rknn_input> inputs(2);
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = joiner_input_attrs_[0].fmt;
    inputs[0].buf = const_cast<float *>(encoder_out);
    inputs[0].size = joiner_input_attrs_[0].n_elems * sizeof(float);

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].fmt = joiner_input_attrs_[1].fmt;
    inputs[1].buf = const_cast<float *>(decoder_out);
    inputs[1].size = joiner_input_attrs_[1].n_elems * sizeof(float);

    std::vector<float> joiner_out(joiner_output_attrs_[0].n_elems);
    rknn_output output;
    output.index = joiner_output_attrs_[0].index;
    output.is_prealloc = 1;
    output.want_float = 1;
    output.size = joiner_out.size() * sizeof(float);
    output.buf = joiner_out.data();

    rknn_context joiner_ctx = 0;
    auto ret = rknn_dup_context(&joiner_ctx_, &joiner_ctx);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to duplicate the joiner ctx");

    ret = rknn_inputs_set(joiner_ctx, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set joiner inputs");

    ret = rknn_run(joiner_ctx, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run joiner");

    ret = rknn_outputs_get(joiner_ctx, 1, &output, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get joiner output");

    rknn_destroy(joiner_ctx);

    return joiner_out;
  }

  int32_t ContextSize() const { return context_size_; }

  int32_t ChunkSize() const { return T_; }

  int32_t ChunkShift() const { return decode_chunk_len_; }

  int32_t VocabSize() const { return vocab_size_; }

  rknn_tensor_attr GetEncoderOutAttr() const {
    return encoder_output_attrs_[0];
  }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &encoder_ctx_);

    InitInputOutputAttrs(encoder_ctx_, config_.debug, &encoder_input_attrs_,
                         &encoder_output_attrs_);

    rknn_custom_string custom_string =
        GetCustomString(encoder_ctx_, config_.debug);

    auto meta = Parse(custom_string, config_.debug);

    if (meta.count("encoder_dims")) {
      SplitStringToIntegers(meta.at("encoder_dims"), ",", false,
                            &encoder_dims_);
    }

    if (meta.count("attention_dims")) {
      SplitStringToIntegers(meta.at("attention_dims"), ",", false,
                            &attention_dims_);
    }

    if (meta.count("num_encoder_layers")) {
      SplitStringToIntegers(meta.at("num_encoder_layers"), ",", false,
                            &num_encoder_layers_);
    }

    if (meta.count("cnn_module_kernels")) {
      SplitStringToIntegers(meta.at("cnn_module_kernels"), ",", false,
                            &cnn_module_kernels_);
    }

    if (meta.count("left_context_len")) {
      SplitStringToIntegers(meta.at("left_context_len"), ",", false,
                            &left_context_len_);
    }

    if (meta.count("T")) {
      T_ = atoi(meta.at("T").c_str());
    }

    if (meta.count("decode_chunk_len")) {
      decode_chunk_len_ = atoi(meta.at("decode_chunk_len").c_str());
    }

    if (meta.count("context_size")) {
      context_size_ = atoi(meta.at("context_size").c_str());
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
    InitContext(model_data, model_data_length, config_.debug, &decoder_ctx_);

    InitInputOutputAttrs(decoder_ctx_, config_.debug, &decoder_input_attrs_,
                         &decoder_output_attrs_);

    if (decoder_input_attrs_[0].type != RKNN_TENSOR_INT64) {
      SHERPA_ONNX_LOGE("Expect int64 for decoder input. Given: %d, %s",
                       decoder_input_attrs_[0].type,
                       get_type_string(decoder_input_attrs_[0].type));
      SHERPA_ONNX_EXIT(-1);
    }

    context_size_ = decoder_input_attrs_[0].dims[1];
    if (config_.debug) {
      SHERPA_ONNX_LOGE("context_size: %d", context_size_);
    }
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, config_.debug, &joiner_ctx_);

    InitInputOutputAttrs(joiner_ctx_, config_.debug, &joiner_input_attrs_,
                         &joiner_output_attrs_);

    vocab_size_ = joiner_output_attrs_[0].dims[1];
    if (config_.debug) {
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
    }
  }

 private:
  OnlineModelConfig config_;
  rknn_context encoder_ctx_ = 0;
  rknn_context decoder_ctx_ = 0;
  rknn_context joiner_ctx_ = 0;

  std::vector<rknn_tensor_attr> encoder_input_attrs_;
  std::vector<rknn_tensor_attr> encoder_output_attrs_;

  std::vector<rknn_tensor_attr> decoder_input_attrs_;
  std::vector<rknn_tensor_attr> decoder_output_attrs_;

  std::vector<rknn_tensor_attr> joiner_input_attrs_;
  std::vector<rknn_tensor_attr> joiner_output_attrs_;

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

OnlineZipformerTransducerModelRknn::~OnlineZipformerTransducerModelRknn() =
    default;

OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<std::vector<uint8_t>>
OnlineZipformerTransducerModelRknn::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>>
OnlineZipformerTransducerModelRknn::RunEncoder(
    std::vector<float> features,
    std::vector<std::vector<uint8_t>> states) const {
  return impl_->RunEncoder(std::move(features), std::move(states));
}

std::vector<float> OnlineZipformerTransducerModelRknn::RunDecoder(
    std::vector<int64_t> decoder_input) const {
  return impl_->RunDecoder(std::move(decoder_input));
}

std::vector<float> OnlineZipformerTransducerModelRknn::RunJoiner(
    const float *encoder_out, const float *decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineZipformerTransducerModelRknn::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelRknn::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineZipformerTransducerModelRknn::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerTransducerModelRknn::VocabSize() const {
  return impl_->VocabSize();
}

rknn_tensor_attr OnlineZipformerTransducerModelRknn::GetEncoderOutAttr() const {
  return impl_->GetEncoderOutAttr();
}

#if __ANDROID_API__ >= 9
template OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformerTransducerModelRknn::OnlineZipformerTransducerModelRknn(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
