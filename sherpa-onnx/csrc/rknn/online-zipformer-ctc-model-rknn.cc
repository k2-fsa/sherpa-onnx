// sherpa-onnx/csrc/rknn/online-zipformer-ctc-model-rknn.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/rknn/online-zipformer-ctc-model-rknn.h"

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

class OnlineZipformerCtcModelRknn::Impl {
 public:
  ~Impl() {
    auto ret = rknn_destroy(ctx_);
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE("Failed to destroy the context");
    }
  }

  explicit Impl(const OnlineModelConfig &config) : config_(config) {
    {
      auto buf = ReadFile(config.zipformer2_ctc.model);
      Init(buf.data(), buf.size());
    }

    int32_t ret = RKNN_SUCC;
    switch (config_.num_threads) {
      case 1:
        ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_AUTO);
        break;
      case 0:
        ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0);
        break;
      case -1:
        ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_1);
        break;
      case -2:
        ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_2);
        break;
      case -3:
        ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0_1);
        break;
      case -4:
        ret = rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0_1_2);
        break;
      default:
        SHERPA_ONNX_LOGE(
            "Valid num_threads for rk npu is 1 (auto), 0 (core 0), -1 (core "
            "1), -2 (core 2), -3 (core 0_1), -4 (core 0_1_2). Given: %d",
            config_.num_threads);
        break;
    }
    if (ret != RKNN_SUCC) {
      SHERPA_ONNX_LOGE(
          "Failed to select npu core to run the model (You can ignore it if "
          "you "
          "are not using RK3588.");
    }
  }

  // TODO(fangjun): Support Android

  std::vector<std::vector<uint8_t>> GetInitStates() const {
    // input_attrs_[0] is for the feature
    // input_attrs_[1:] is for states
    // so we use -1 here
    std::vector<std::vector<uint8_t>> states(input_attrs_.size() - 1);

    int32_t i = -1;
    for (auto &attr : input_attrs_) {
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

  std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>> Run(
      std::vector<float> features,
      std::vector<std::vector<uint8_t>> states) const {
    std::vector<rknn_input> inputs(input_attrs_.size());

    for (int32_t i = 0; i < static_cast<int32_t>(inputs.size()); ++i) {
      auto &input = inputs[i];
      auto &attr = input_attrs_[i];
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

    std::vector<float> out(output_attrs_[0].n_elems);

    // Note(fangjun): We can reuse the memory from input argument `states`
    // auto next_states = GetInitStates();
    auto &next_states = states;

    std::vector<rknn_output> outputs(output_attrs_.size());
    for (int32_t i = 0; i < outputs.size(); ++i) {
      auto &output = outputs[i];
      auto &attr = output_attrs_[i];
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
        output.size = out.size() * sizeof(float);
        output.buf = reinterpret_cast<void *>(out.data());
      } else {
        output.size = next_states[i - 1].size();
        output.buf = reinterpret_cast<void *>(next_states[i - 1].data());
      }
    }

    auto ret = rknn_inputs_set(ctx_, inputs.size(), inputs.data());
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to set inputs");

    ret = rknn_run(ctx_, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to run the model");

    ret = rknn_outputs_get(ctx_, outputs.size(), outputs.data(), nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get model output");

    for (int32_t i = 0; i < next_states.size(); ++i) {
      const auto &attr = input_attrs_[i + 1];
      if (attr.n_dims == 4) {
        // TODO(fangjun): The transpose is copied from
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

    return {std::move(out), std::move(next_states)};
  }

  int32_t ChunkSize() const { return T_; }

  int32_t ChunkShift() const { return decode_chunk_len_; }

  int32_t VocabSize() const { return vocab_size_; }

  rknn_tensor_attr GetOutAttr() const { return output_attrs_[0]; }

 private:
  void Init(void *model_data, size_t model_data_length) {
    auto ret = rknn_init(&ctx_, model_data, model_data_length, 0, nullptr);
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to init model '%s'",
                           config_.zipformer2_ctc.model.c_str());

    if (config_.debug) {
      rknn_sdk_version v;
      ret = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &v, sizeof(v));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get rknn sdk version");

      SHERPA_ONNX_LOGE("sdk api version: %s, driver version: %s", v.api_version,
                       v.drv_version);
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get I/O information for the model");

    if (config_.debug) {
      SHERPA_ONNX_LOGE("model: %d inputs, %d outputs",
                       static_cast<int32_t>(io_num.n_input),
                       static_cast<int32_t>(io_num.n_output));
    }

    input_attrs_.resize(io_num.n_input);
    output_attrs_.resize(io_num.n_output);

    int32_t i = 0;
    for (auto &attr : input_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for model input %d", i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : input_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Model inputs info----------\n%s",
                       os.str().c_str());
    }

    i = 0;
    for (auto &attr : output_attrs_) {
      memset(&attr, 0, sizeof(attr));
      attr.index = i;
      ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
      SHERPA_ONNX_RKNN_CHECK(ret, "Failed to get attr for model output %d", i);
      i += 1;
    }

    if (config_.debug) {
      std::ostringstream os;
      std::string sep;
      for (auto &attr : output_attrs_) {
        os << sep << ToString(attr);
        sep = "\n";
      }
      SHERPA_ONNX_LOGE("\n----------Model outputs info----------\n%s",
                       os.str().c_str());
    }

    rknn_custom_string custom_string;
    ret = rknn_query(ctx_, RKNN_QUERY_CUSTOM_STRING, &custom_string,
                     sizeof(custom_string));
    SHERPA_ONNX_RKNN_CHECK(ret, "Failed to read custom string from the model");
    if (config_.debug) {
      SHERPA_ONNX_LOGE("customs string: %s", custom_string.string);
    }
    auto meta = Parse(custom_string);

    if (config_.debug) {
      for (const auto &p : meta) {
        SHERPA_ONNX_LOGE("%s: %s", p.first.c_str(), p.second.c_str());
      }
    }

    if (meta.count("T")) {
      T_ = atoi(meta.at("T").c_str());
    }

    if (meta.count("decode_chunk_len")) {
      decode_chunk_len_ = atoi(meta.at("decode_chunk_len").c_str());
    }

    vocab_size_ = output_attrs_[0].dims[2];

    if (config_.debug) {
#if __OHOS__
      SHERPA_ONNX_LOGE("T: %{public}d", T_);
      SHERPA_ONNX_LOGE("decode_chunk_len_: %{public}d", decode_chunk_len_);
      SHERPA_ONNX_LOGE("vocab_size: %{public}d", vocab_size);
#else
      SHERPA_ONNX_LOGE("T: %d", T_);
      SHERPA_ONNX_LOGE("decode_chunk_len_: %d", decode_chunk_len_);
      SHERPA_ONNX_LOGE("vocab_size: %d", vocab_size_);
#endif
    }

    if (T_ == 0) {
      SHERPA_ONNX_LOGE(
          "Invalid T. Please use the script from icefall to export your model");
      SHERPA_ONNX_EXIT(-1);
    }

    if (decode_chunk_len_ == 0) {
      SHERPA_ONNX_LOGE(
          "Invalid decode_chunk_len. Please use the script from icefall to "
          "export your model");
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  OnlineModelConfig config_;
  rknn_context ctx_ = 0;

  std::vector<rknn_tensor_attr> input_attrs_;
  std::vector<rknn_tensor_attr> output_attrs_;

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;
  int32_t vocab_size_ = 0;
};

OnlineZipformerCtcModelRknn::~OnlineZipformerCtcModelRknn() = default;

OnlineZipformerCtcModelRknn::OnlineZipformerCtcModelRknn(
    const OnlineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OnlineZipformerCtcModelRknn::OnlineZipformerCtcModelRknn(
    Manager *mgr, const OnlineModelConfig &config)
    : impl_(std::make_unique<OnlineZipformerCtcModelRknn>(mgr, config)) {}

std::vector<std::vector<uint8_t>> OnlineZipformerCtcModelRknn::GetInitStates()
    const {
  return impl_->GetInitStates();
}

std::pair<std::vector<float>, std::vector<std::vector<uint8_t>>>
OnlineZipformerCtcModelRknn::Run(
    std::vector<float> features,
    std::vector<std::vector<uint8_t>> states) const {
  return impl_->Run(std::move(features), std::move(states));
}

int32_t OnlineZipformerCtcModelRknn::ChunkSize() const {
  return impl_->ChunkSize();
}

int32_t OnlineZipformerCtcModelRknn::ChunkShift() const {
  return impl_->ChunkShift();
}

int32_t OnlineZipformerCtcModelRknn::VocabSize() const {
  return impl_->VocabSize();
}

rknn_tensor_attr OnlineZipformerCtcModelRknn::GetOutAttr() const {
  return impl_->GetOutAttr();
}

#if __ANDROID_API__ >= 9
template OnlineZipformerCtcModelRknn::OnlineZipformerCtcModelRknn(
    AAssetManager *mgr, const OnlineModelConfig &config);
#endif

#if __OHOS__
template OnlineZipformerCtcModelRknn::OnlineZipformerCtcModelRknn(
    NativeResourceManager *mgr, const OnlineModelConfig &config);
#endif

}  // namespace sherpa_onnx
