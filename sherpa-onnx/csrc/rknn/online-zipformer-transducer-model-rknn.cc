// sherpa-onnx/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.h"

#include <algorithm>
#include <cassert>
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
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static std::string ToString(const rknn_tensor_attr &attr) {
  std::ostringstream os;
  os << "{";
  os << attr.index;
  os << ", name: " << attr.name;
  os << ", shape: (";
  std::string sep;
  for (int32_t i = 0; i < static_cast<int32_t>(attr.n_dims); ++i) {
    os << sep << attr.dims[i];
    sep = ",";
  }
  os << ")";
  os << ", n_elems: " << attr.n_elems;
  os << ", size: " << attr.size;
  os << ", fmt: " << get_format_string(attr.fmt);
  os << ", type: " << get_type_string(attr.type);
  os << ", pass_through: " << (attr.pass_through ? "true" : "false");
  os << "}";
  return os.str();
}

static std::unordered_map<std::string, std::string> Parse(
    const rknn_custom_string &custom_string) {
  std::unordered_map<std::string, std::string> ans;
  std::vector<std::string> fields;
  SplitStringToVector(custom_string.string, ";", false, &fields);

  std::vector<std::string> tmp;
  for (const auto &f : fields) {
    SplitStringToVector(f, "=", false, &tmp);
    if (tmp.size() != 2) {
      SHERPA_ONNX_LOGE("Invalid custom string %s for %s", custom_string.string,
                       f.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    ans[tmp[0]] = tmp[1];
  }

  return ans;
}

class OnlineZipformerTransducerModelRknn::Impl {
 public:
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
  }

  std::vector<std::vector<uint8_t>> GetEncoderInitStates() const { return {}; }

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> RunEncoder(
      const std::vector<float> &features,
      std::vector<std::vector<uint8_t>> states) const {
    return {};
  }

  std::vector<float> RunDecoder(
      const std::vector<int64_t> &decoder_input) const {
    return {};
  }

  std::vector<float> RunJoiner(const std::vector<float> &encoder_out,
                               const std::vector<float> &decoder_out) const {
    return {};
  }

  int32_t ContextSize() const { return 0; }

  int32_t ChunkSize() const { return 0; }

  int32_t ChunkShift() const { return 0; }

  int32_t VocabSize() const { return 0; }

 private:
  void InitEncoder(void *model_data, size_t model_data_length) {
    auto ret =
        rknn_init(&encoder_ctx_, model_data, model_data_length, 0, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to init encoder '%s'",
                           config_.transducer.encoder.c_str());

    if (config_.debug) {
      rknn_sdk_version v;
      ret = rknn_query(encoder_ctx_, RKNN_QUERY_SDK_VERSION, &v, sizeof(v));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get rknn sdk version");

      SHERPA_ONNX_LOGE("sdk api version: %s, driver version: %s", v.api_version,
                       v.drv_version);
    }

    rknn_input_output_num io_num;
    ret = rknn_query(encoder_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num,
                     sizeof(io_num));
    SHERPA_ONNX_RKNN_CHECK(ret,
                           "Failed to get I/O information for the encoder");

    if (config_.debug) {
      SHERPA_ONNX_LOGE("encoder: %d inputs, %d outputs",
                       static_cast<int32_t>(io_num.n_input),
                       static_cast<int32_t>(io_num.n_output));
    }

    encoder_input_attrs_.resize(io_num.n_input);
    encoder_output_attrs_.resize(io_num.n_output);

    int32_t i = 0;
    for (auto &attr : encoder_input_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret =
          rknn_query(encoder_ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for encoder input %d", i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : encoder_input_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Encoder inputs info----------\n%s",
                       os.str().c_str());
    }

    i = 0;
    for (auto &attr : encoder_output_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret =
          rknn_query(encoder_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for encoder output %d",
                             i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : encoder_output_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Encoder outputs info----------\n%s",
                       os.str().c_str());
    }

    rknn_custom_string custom_string;
    ret = rknn_query(encoder_ctx_, RKNN_QUERY_CUSTOM_STRING, &custom_string,
                     sizeof(custom_string));
    SHERPA_ONNX_RKNN_CHECK(
        ret, "Failed to read custom string from the encoder model");
    if (config_.debug) {
      SHERPA_ONNX_LOGE("customs string: %s", custom_string.string);
    }
    auto meta = Parse(custom_string);

    for (const auto &p : meta) {
      SHERPA_ONNX_LOGE("%s: %s", p.first.c_str(), p.second.c_str());
    }

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
      SHERPA_ONNX_LOGE("context_size: %{public}d", context_size_);
#else
      SHERPA_ONNX_LOGE("T: %d", T_);
      SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);
      SHERPA_ONNX_LOGE("context_size: %d", context_size_);
#endif
    }
  }

  void InitDecoder(void *model_data, size_t model_data_length) {
    auto ret =
        rknn_init(&decoder_ctx_, model_data, model_data_length, 0, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to init decoder '%s'",
                           config_.transducer.decoder.c_str());

    rknn_input_output_num io_num;
    ret = rknn_query(decoder_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num,
                     sizeof(io_num));
    SHERPA_ONNX_RKNN_CHECK(ret,
                           "Failed to get I/O information for the decoder");

    if (config_.debug) {
      SHERPA_ONNX_LOGE("decoder: %d inputs, %d outputs",
                       static_cast<int32_t>(io_num.n_input),
                       static_cast<int32_t>(io_num.n_output));
    }

    decoder_input_attrs_.resize(io_num.n_input);
    decoder_output_attrs_.resize(io_num.n_output);

    int32_t i = 0;
    for (auto &attr : decoder_input_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret =
          rknn_query(decoder_ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for decoder input %d", i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : decoder_input_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Decoder inputs info----------\n%s",
                       os.str().c_str());
    }

    i = 0;
    for (auto &attr : decoder_output_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret =
          rknn_query(decoder_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for decoder output %d",
                             i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : decoder_output_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Decoder outputs info----------\n%s",
                       os.str().c_str());
    }
  }

  void InitJoiner(void *model_data, size_t model_data_length) {
    auto ret =
        rknn_init(&joiner_ctx_, model_data, model_data_length, 0, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to init joiner '%s'",
                           config_.transducer.joiner.c_str());

    rknn_input_output_num io_num;
    ret =
        rknn_query(joiner_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get I/O information for the joiner");

    if (config_.debug) {
      SHERPA_ONNX_LOGE("joiner: %d inputs, %d outputs",
                       static_cast<int32_t>(io_num.n_input),
                       static_cast<int32_t>(io_num.n_output));
    }

    joiner_input_attrs_.resize(io_num.n_input);
    joiner_output_attrs_.resize(io_num.n_output);

    int32_t i = 0;
    for (auto &attr : joiner_input_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret = rknn_query(joiner_ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for joiner input %d", i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : joiner_input_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Joiner inputs info----------\n%s",
                       os.str().c_str());
    }

    i = 0;
    for (auto &attr : joiner_output_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret =
          rknn_query(joiner_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for joiner output %d", i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : joiner_output_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Joiner outputs info----------\n%s",
                       os.str().c_str());
    }

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
    : impl_(std::make_unique<OnlineZipformerTransducerModelRknn>(mgr, config)) {
}

std::vector<std::vector<uint8_t>>
OnlineZipformerTransducerModelRknn::GetEncoderInitStates() const {
  return impl_->GetEncoderInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>>
OnlineZipformerTransducerModelRknn::RunEncoder(
    const std::vector<float> &features,
    std::vector<std::vector<uint8_t>> states) const {
  return impl_->RunEncoder(features, std::move(states));
}

std::vector<float> OnlineZipformerTransducerModelRknn::RunDecoder(
    const std::vector<int64_t> &decoder_input) const {
  return impl_->RunDecoder(decoder_input);
}

std::vector<float> OnlineZipformerTransducerModelRknn::RunJoiner(
    const std::vector<float> &encoder_out,
    const std::vector<float> &decoder_out) const {
  return impl_->RunJoiner(encoder_out, decoder_out);
}

int32_t OnlineZipformerTransducerModelRknn::ContextSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelRknn::ChunkSize() const {
  return impl_->ContextSize();
}

int32_t OnlineZipformerTransducerModelRknn::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerTransducerModelRknn::VocabSize() const {
  return impl_->VocabSize();
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
