// sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.cc
//
// Copyright (c)  2026  Ceva Inc

#include "sherpa-onnx/csrc/offline-speech-denoiser-dpdfnet-model.h"

#include <algorithm>
#include <array>
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
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

std::vector<int64_t> GetInputShape(Ort::Session *sess, size_t index) {
  return sess->GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<int64_t> GetOutputShape(Ort::Session *sess, size_t index) {
  return sess->GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
}

}  // namespace

class OfflineSpeechDenoiserDpdfNetModel::Impl {
 public:
  explicit Impl(const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(config.dpdfnet.model);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.dpdfnet.model);
    Init(buf.data(), buf.size());
  }

  Ort::Value GetInitState() {
    Ort::Value state = Ort::Value::CreateTensor<float>(
        allocator_, meta_.state_shape.data(), meta_.state_shape.size());

    auto *p = state.GetTensorMutableData<float>();
    std::fill_n(p, meta_.state_size, 0.f);
    std::copy(meta_.erb_norm_init.begin(), meta_.erb_norm_init.end(), p);
    std::copy(meta_.spec_norm_init.begin(), meta_.spec_norm_init.end(),
              p + meta_.erb_norm_state_size);

    return state;
  }

  std::pair<Ort::Value, Ort::Value> Run(Ort::Value x, Ort::Value state) const {
    std::array<Ort::Value, 2> inputs{std::move(x), std::move(state)};

    auto out =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());

    return {std::move(out[0]), std::move(out[1])};
  }

  const OfflineSpeechDenoiserDpdfNetModelMetaData &GetMetaData() const {
    return meta_;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;  // used in the macros below

    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
    if (model_type != "dpdfnet") {
      SHERPA_ONNX_LOGE("Expect model type 'dpdfnet'. Given: '%s'",
                       model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA_STR(meta_.profile, "profile");
    SHERPA_ONNX_READ_META_DATA(meta_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_.n_fft, "n_fft");
    SHERPA_ONNX_READ_META_DATA(meta_.hop_length, "hop_length");
    SHERPA_ONNX_READ_META_DATA(meta_.window_length, "window_length");
    int32_t normalized = 0;
    int32_t center = 1;
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(normalized, "normalized", 0);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(center, "center", 1);
    SHERPA_ONNX_READ_META_DATA_STR(meta_.window_type, "window_type");
    SHERPA_ONNX_READ_META_DATA_STR(meta_.pad_mode, "pad_mode");
    SHERPA_ONNX_READ_META_DATA(meta_.freq_bins, "freq_bins");
    SHERPA_ONNX_READ_META_DATA(meta_.erb_bins, "erb_bins");
    SHERPA_ONNX_READ_META_DATA(meta_.spec_bins, "spec_bins");
    SHERPA_ONNX_READ_META_DATA(meta_.state_size, "state_size");
    SHERPA_ONNX_READ_META_DATA(meta_.erb_norm_state_size,
                               "erb_norm_state_size");
    SHERPA_ONNX_READ_META_DATA(meta_.spec_norm_state_size,
                               "spec_norm_state_size");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_.erb_norm_init, "erb_norm_init");
    SHERPA_ONNX_READ_META_DATA_VEC_FLOAT(meta_.spec_norm_init,
                                         "spec_norm_init");

    if (normalized > 1 || center > 1) {
      SHERPA_ONNX_LOGE(
          "Invalid boolean metadata values. normalized=%d, center=%d.",
          normalized, center);
      SHERPA_ONNX_EXIT(-1);
    }

    meta_.normalized = normalized != 0;
    meta_.center = center != 0;

    if (meta_.sample_rate <= 0 || meta_.n_fft <= 0 || meta_.hop_length <= 0 ||
        meta_.window_length <= 0 || meta_.freq_bins <= 1 ||
        meta_.erb_bins <= 0 || meta_.spec_bins <= 0 || meta_.state_size <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid DPDFNet metadata. sample_rate=%d, n_fft=%d, "
          "hop_length=%d, window_length=%d, freq_bins=%d, erb_bins=%d, "
          "spec_bins=%d, state_size=%d.",
          meta_.sample_rate, meta_.n_fft, meta_.hop_length, meta_.window_length,
          meta_.freq_bins, meta_.erb_bins, meta_.spec_bins, meta_.state_size);
      SHERPA_ONNX_EXIT(-1);
    }

    if (input_names_.size() != 2 || output_names_.size() != 2) {
      SHERPA_ONNX_LOGE(
          "Expect the dpdfnet model to have 2 inputs and 2 outputs. "
          "Got %zu inputs and %zu outputs.",
          input_names_.size(), output_names_.size());
      SHERPA_ONNX_EXIT(-1);
    }

    auto spec_shape = GetInputShape(sess_.get(), 0);
    auto state_shape = GetInputShape(sess_.get(), 1);
    auto out_spec_shape = GetOutputShape(sess_.get(), 0);
    auto out_state_shape = GetOutputShape(sess_.get(), 1);

    if (spec_shape.size() != 4 || state_shape.size() != 1 ||
        out_spec_shape.size() != 4 || out_state_shape.size() != 1) {
      SHERPA_ONNX_LOGE(
          "Unexpected dpdfnet ONNX signature. Expected "
          "(spec:[B,T,F,2], state:[S]) -> (spec_e:[B,T,F,2], state_out:[S]). "
          "Got spec ndim=%d, state ndim=%d, out_spec ndim=%d, out_state "
          "ndim=%d.",
          static_cast<int32_t>(spec_shape.size()),
          static_cast<int32_t>(state_shape.size()),
          static_cast<int32_t>(out_spec_shape.size()),
          static_cast<int32_t>(out_state_shape.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    const int64_t freq_bins = spec_shape[2];
    const int64_t complex_dim = spec_shape[3];
    const int64_t state_size = state_shape[0];

    if (freq_bins <= 1 || complex_dim != 2 || state_size <= 0) {
      SHERPA_ONNX_LOGE(
          "Unsupported dpdfnet model shapes. spec ndim=%d, state ndim=%d, "
          "freq_bins=%d, complex_dim=%d, state_size=%d.",
          static_cast<int32_t>(spec_shape.size()),
          static_cast<int32_t>(state_shape.size()),
          static_cast<int32_t>(freq_bins), static_cast<int32_t>(complex_dim),
          static_cast<int32_t>(state_size));
      SHERPA_ONNX_EXIT(-1);
    }

    meta_.spec_shape = std::move(spec_shape);
    meta_.state_shape = std::move(state_shape);

    if (meta_.freq_bins != freq_bins) {
      SHERPA_ONNX_LOGE(
          "Mismatch between metadata and ONNX graph for freq_bins. "
          "metadata=%d, graph=%d.",
          meta_.freq_bins, static_cast<int32_t>(freq_bins));
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta_.n_fft != static_cast<int32_t>((freq_bins - 1) * 2)) {
      SHERPA_ONNX_LOGE(
          "Mismatch between metadata and ONNX graph for n_fft. metadata=%d, "
          "graph=%d.",
          meta_.n_fft, static_cast<int32_t>((freq_bins - 1) * 2));
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta_.state_size != state_size) {
      SHERPA_ONNX_LOGE(
          "Mismatch between metadata and ONNX graph for state_size. "
          "metadata=%d, graph=%d.",
          meta_.state_size, static_cast<int32_t>(state_size));
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta_.erb_norm_state_size !=
        static_cast<int32_t>(meta_.erb_norm_init.size())) {
      SHERPA_ONNX_LOGE(
          "Mismatch between erb_norm_state_size (%d) and erb_norm_init size "
          "(%zu).",
          meta_.erb_norm_state_size, meta_.erb_norm_init.size());
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta_.spec_norm_state_size !=
        static_cast<int32_t>(meta_.spec_norm_init.size())) {
      SHERPA_ONNX_LOGE(
          "Mismatch between spec_norm_state_size (%d) and spec_norm_init size "
          "(%zu).",
          meta_.spec_norm_state_size, meta_.spec_norm_init.size());
      SHERPA_ONNX_EXIT(-1);
    }

    const int32_t init_prefix_state_size =
        meta_.erb_norm_state_size + meta_.spec_norm_state_size;
    if (meta_.erb_norm_state_size <= 0 || meta_.spec_norm_state_size <= 0) {
      SHERPA_ONNX_LOGE(
          "Invalid normalization state sizes in the metadata. "
          "erb_norm_state_size=%d, spec_norm_state_size=%d.",
          meta_.erb_norm_state_size, meta_.spec_norm_state_size);
      SHERPA_ONNX_EXIT(-1);
    }

    if (meta_.state_size < init_prefix_state_size) {
      SHERPA_ONNX_LOGE(
          "The dpdfnet state tensor is too small: %d. It must be at least %d.",
          meta_.state_size, init_prefix_state_size);
      SHERPA_ONNX_EXIT(-1);
    }

    if (out_spec_shape[2] != freq_bins || out_spec_shape[3] != 2 ||
        out_state_shape[0] != state_size) {
      SHERPA_ONNX_LOGE(
          "Unexpected dpdfnet output shapes. out_spec[2]=%d, out_spec[3]=%d, "
          "out_state[0]=%d, expected freq_bins=%d, complex_dim=2, "
          "state_size=%d.",
          static_cast<int32_t>(out_spec_shape[2]),
          static_cast<int32_t>(out_spec_shape[3]),
          static_cast<int32_t>(out_state_shape[0]),
          static_cast<int32_t>(freq_bins), static_cast<int32_t>(state_size));
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.debug) {
      std::ostringstream os;
      os << "---dpdfnet model---\n";
      PrintModelMetadata(os, meta_data);
      os << "input names:\n";
      for (int32_t i = 0; i != static_cast<int32_t>(input_names_.size()); ++i) {
        os << i << " " << input_names_[i] << "\n";
      }

      os << "output names:\n";
      for (int32_t i = 0; i != static_cast<int32_t>(output_names_.size());
           ++i) {
        os << i << " " << output_names_[i] << "\n";
      }

      os << "spec shape: ";
      for (auto d : meta_.spec_shape) {
        os << d << " ";
      }
      os << "\nstate shape: ";
      for (auto d : meta_.state_shape) {
        os << d << " ";
      }
      os << "\nprofile: " << meta_.profile;
      os << "\nsample_rate: " << meta_.sample_rate;
      os << "\nn_fft: " << meta_.n_fft;
      os << "\nfreq_bins: " << meta_.freq_bins;
      os << "\nerb_bins: " << meta_.erb_bins;
      os << "\nspec_bins: " << meta_.spec_bins;
      os << "\nstate_size: " << meta_.state_size;
      os << "\nnormalized: " << static_cast<int32_t>(meta_.normalized);
      os << "\ncenter: " << static_cast<int32_t>(meta_.center);
      os << "\n";

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
  }

 private:
  OfflineSpeechDenoiserModelConfig config_;
  OfflineSpeechDenoiserDpdfNetModelMetaData meta_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

OfflineSpeechDenoiserDpdfNetModel::~OfflineSpeechDenoiserDpdfNetModel() =
    default;  // NOLINT

OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

Ort::Value OfflineSpeechDenoiserDpdfNetModel::GetInitState() const {
  return impl_->GetInitState();
}

std::pair<Ort::Value, Ort::Value> OfflineSpeechDenoiserDpdfNetModel::Run(
    Ort::Value x, Ort::Value state) const {
  return impl_->Run(std::move(x), std::move(state));
}

const OfflineSpeechDenoiserDpdfNetModelMetaData &
OfflineSpeechDenoiserDpdfNetModel::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    AAssetManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

#if __OHOS__
template OfflineSpeechDenoiserDpdfNetModel::OfflineSpeechDenoiserDpdfNetModel(
    NativeResourceManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

}  // namespace sherpa_onnx
