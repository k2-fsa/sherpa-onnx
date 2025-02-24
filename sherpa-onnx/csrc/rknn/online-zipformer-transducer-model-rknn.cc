// sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.cc
//
// Copyright (c)  2023  Xiaomi Corporation

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
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

// chw -> hwc
static void Transpose(const float *src, int32_t n, int32_t channel,
                      int32_t height, int32_t width, float *dst) {
  for (int32_t i = 0; i < n; ++i) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        for (int32_t c = 0; c < channel; ++c) {
          // dst[h, w, c] = src[c, h, w]
          dst[i * height * width * channel + h * width * channel + w * channel +
              c] = src[i * height * width * channel + c * height * width +
                       h * width + w];
        }
      }
    }
  }
}

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
    ans[std::move(tmp[0])] = std::move(tmp[1]);
  }

  return ans;
}

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

    // Now select which core to run for RK3588
    int32_t ret_encoder = RKNN_SUCC;
    int32_t ret_decoder = RKNN_SUCC;
    int32_t ret_joiner = RKNN_SUCC;
    switch (config_.num_threads) {
      case 1:
        ret_encoder = rknn_set_core_mask(encoder_ctx_, RKNN_NPU_CORE_AUTO);
        ret_decoder = rknn_set_core_mask(decoder_ctx_, RKNN_NPU_CORE_AUTO);
        ret_joiner = rknn_set_core_mask(joiner_ctx_, RKNN_NPU_CORE_AUTO);
        break;
      case 0:
        ret_encoder = rknn_set_core_mask(encoder_ctx_, RKNN_NPU_CORE_0);
        ret_decoder = rknn_set_core_mask(decoder_ctx_, RKNN_NPU_CORE_0);
        ret_joiner = rknn_set_core_mask(joiner_ctx_, RKNN_NPU_CORE_0);
        break;
      case -1:
        ret_encoder = rknn_set_core_mask(encoder_ctx_, RKNN_NPU_CORE_1);
        ret_decoder = rknn_set_core_mask(decoder_ctx_, RKNN_NPU_CORE_1);
        ret_joiner = rknn_set_core_mask(joiner_ctx_, RKNN_NPU_CORE_1);
        break;
      case -2:
        ret_encoder = rknn_set_core_mask(encoder_ctx_, RKNN_NPU_CORE_2);
        ret_decoder = rknn_set_core_mask(decoder_ctx_, RKNN_NPU_CORE_2);
        ret_joiner = rknn_set_core_mask(joiner_ctx_, RKNN_NPU_CORE_2);
        break;
      case -3:
        ret_encoder = rknn_set_core_mask(encoder_ctx_, RKNN_NPU_CORE_0_1);
        ret_decoder = rknn_set_core_mask(decoder_ctx_, RKNN_NPU_CORE_0_1);
        ret_joiner = rknn_set_core_mask(joiner_ctx_, RKNN_NPU_CORE_0_1);
        break;
      case -4:
        ret_encoder = rknn_set_core_mask(encoder_ctx_, RKNN_NPU_CORE_0_1_2);
        ret_decoder = rknn_set_core_mask(decoder_ctx_, RKNN_NPU_CORE_0_1_2);
        ret_joiner = rknn_set_core_mask(joiner_ctx_, RKNN_NPU_CORE_0_1_2);
        break;
      default:
        SHERPA_ONNX_LOGE(
            "Valid num_threads for rk npu is 1 (auto), 0 (core 0), -1 (core "
            "1), -2 (core 2), -3 (core 0_1), -4 (core 0_1_2). Given: %d",
            config_.num_threads);
        break;
    }
    if (ret_encoder != RKNN_SUCC) {
      SHERPA_ONNX_LOGE(
          "Failed to select npu core to run encoder (You can ignore it if you "
          "are not using RK3588.");
    }

    if (ret_decoder != RKNN_SUCC) {
      SHERPA_ONNX_LOGE(
          "Failed to select npu core to run decoder (You can ignore it if you "
          "are not using RK3588.");
    }

    if (ret_decoder != RKNN_SUCC) {
      SHERPA_ONNX_LOGE(
          "Failed to select npu core to run joiner (You can ignore it if you "
          "are not using RK3588.");
    }
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
      std::vector<float> features,
      std::vector<std::vector<uint8_t>> states) const {
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

    auto ret = rknn_inputs_set(encoder_ctx_, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set encoder inputs");

    ret = rknn_run(encoder_ctx_, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run encoder");

    ret =
        rknn_outputs_get(encoder_ctx_, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get encoder output");

    for (int32_t i = 0; i < next_states.size(); ++i) {
      const auto &attr = encoder_input_attrs_[i + 1];
      if (attr.n_dims == 4) {
        // TODO(fangjun): The transpose is copied from
        // https://github.com/airockchip/rknn_model_zoo/blob/main/examples/zipformer/cpp/process.cc#L22
        // I don't understand why we need to do that.
        std::vector<uint8_t> dst(next_states[i].size());
        int32_t n = attr.dims[0];
        int32_t h = attr.dims[1];
        int32_t w = attr.dims[2];
        int32_t c = attr.dims[3];
        Transpose(reinterpret_cast<const float *>(next_states[i].data()), n, c,
                  h, w, reinterpret_cast<float *>(dst.data()));
        next_states[i] = std::move(dst);
      }
    }

    return {std::move(encoder_out), std::move(next_states)};
  }

  std::vector<float> RunDecoder(std::vector<int64_t> decoder_input) const {
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

    auto ret = rknn_inputs_set(decoder_ctx_, 1, &input);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set decoder inputs");

    ret = rknn_run(decoder_ctx_, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run decoder");

    ret = rknn_outputs_get(decoder_ctx_, 1, &output, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get decoder output");

    return decoder_out;
  }

  std::vector<float> RunJoiner(const float *encoder_out,
                               const float *decoder_out) const {
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

    auto ret = rknn_inputs_set(joiner_ctx_, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set joiner inputs");

    ret = rknn_run(joiner_ctx_, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run joiner");

    ret = rknn_outputs_get(joiner_ctx_, 1, &output, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get joiner output");

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

    if (io_num.n_input != 1) {
      SHERPA_ONNX_LOGE("Expect only 1 decoder input. Given %d",
                       static_cast<int32_t>(io_num.n_input));
      SHERPA_ONNX_EXIT(-1);
    }

    if (io_num.n_output != 1) {
      SHERPA_ONNX_LOGE("Expect only 1 decoder output. Given %d",
                       static_cast<int32_t>(io_num.n_output));
      SHERPA_ONNX_EXIT(-1);
    }

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

    if (decoder_input_attrs_[0].type != RKNN_TENSOR_INT64) {
      SHERPA_ONNX_LOGE("Expect int64 for decoder input. Given: %d, %s",
                       decoder_input_attrs_[0].type,
                       get_type_string(decoder_input_attrs_[0].type));
      SHERPA_ONNX_EXIT(-1);
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
