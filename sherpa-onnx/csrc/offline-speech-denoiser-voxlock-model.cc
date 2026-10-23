// sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model.cc
//
// Copyright (c)  2026  Arbaz Siddiqui

#include "sherpa-onnx/csrc/offline-speech-denoiser-voxlock-model.h"

#include <array>
#include <cmath>
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

// Properties that both the enroll and the chunk model must agree on
struct VoxLockSharedMetaData {
  int32_t version = 0;
  int32_t sample_rate = 0;
  int32_t n_fft = 0;
  int32_t hop_length = 0;
  int32_t window_length = 0;
  std::string window_type;
};

VoxLockSharedMetaData ReadSharedMetaData(const Ort::ModelMetadata &meta_data,
                                         const std::string &expected_role,
                                         const std::string &filename) {
  Ort::AllocatorWithDefaultOptions allocator;  // used in the macros below

  std::string model_type;
  SHERPA_ONNX_READ_META_DATA_STR(model_type, "model_type");
  if (model_type != "voxlock") {
    SHERPA_ONNX_LOGE("Expect model type 'voxlock' in '%s'. Given: '%s'",
                     filename.c_str(), model_type.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  std::string role;
  SHERPA_ONNX_READ_META_DATA_STR(role, "model_role");
  if (role != expected_role) {
    SHERPA_ONNX_LOGE(
        "Expect a voxlock %s model in '%s'. Given a voxlock %s model. Did you "
        "swap the enroll and chunk models?",
        expected_role.c_str(), filename.c_str(), role.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  VoxLockSharedMetaData ans;
  SHERPA_ONNX_READ_META_DATA(ans.version, "version");
  SHERPA_ONNX_READ_META_DATA(ans.sample_rate, "sample_rate");
  SHERPA_ONNX_READ_META_DATA(ans.n_fft, "n_fft");
  SHERPA_ONNX_READ_META_DATA(ans.hop_length, "hop_length");
  SHERPA_ONNX_READ_META_DATA(ans.window_length, "window_length");
  SHERPA_ONNX_READ_META_DATA_STR(ans.window_type, "window_type");

  return ans;
}

}  // namespace

class OfflineSpeechDenoiserVoxLockModel::Impl {
 public:
  explicit Impl(const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(config.voxlock.enroll_model);
      InitEnroll(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(config.voxlock.chunk_model);
      InitChunk(buf.data(), buf.size());
    }
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.voxlock.enroll_model);
      InitEnroll(buf.data(), buf.size());
    }

    {
      auto buf = ReadFile(mgr, config.voxlock.chunk_model);
      InitChunk(buf.data(), buf.size());
    }
  }

  std::pair<Ort::Value, Ort::Value> Enroll(const float *samples,
                                           int32_t n) const {
    if (n != meta_.voice_sample_seconds * meta_.sample_rate) {
      SHERPA_ONNX_LOGE(
          "The voxlock voice sample must have exactly %d samples (%d seconds "
          "at %d Hz). Given: %d",
          meta_.voice_sample_seconds * meta_.sample_rate,
          meta_.voice_sample_seconds, meta_.sample_rate, n);
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<float> x(samples, samples + n);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> x_shape{1, n};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    auto out = enroll_sess_->Run({}, enroll_input_names_ptr_.data(), &x_tensor,
                                 1, enroll_output_names_ptr_.data(),
                                 enroll_output_names_ptr_.size());

    return {std::move(out[0]), std::move(out[1])};
  }

  States GetInitStates() {
    Ort::Value h = Ort::Value::CreateTensor<float>(
        allocator_, meta_.state_shape.data(), meta_.state_shape.size());

    Ort::Value c = Ort::Value::CreateTensor<float>(
        allocator_, meta_.state_shape.data(), meta_.state_shape.size());

    Fill<float>(&h, 0);
    Fill<float>(&c, 0);

    States states;
    states.reserve(2);
    states.push_back(std::move(h));
    states.push_back(std::move(c));

    return states;
  }

  std::pair<Ort::Value, States> Run(Ort::Value wav, Ort::Value enroll_mag,
                                    Ort::Value context, States states) const {
    std::vector<Ort::Value> inputs;
    inputs.reserve(3 + states.size());
    inputs.push_back(std::move(wav));
    inputs.push_back(std::move(enroll_mag));
    inputs.push_back(std::move(context));
    for (auto &s : states) {
      inputs.push_back(std::move(s));
    }

    auto out = chunk_sess_->Run(
        {}, chunk_input_names_ptr_.data(), inputs.data(), inputs.size(),
        chunk_output_names_ptr_.data(), chunk_output_names_ptr_.size());

    States next_states;
    next_states.reserve(out.size() - 1);
    for (int32_t k = 1; k < static_cast<int32_t>(out.size()); ++k) {
      next_states.push_back(std::move(out[k]));
    }

    return {std::move(out[0]), std::move(next_states)};
  }

  const OfflineSpeechDenoiserVoxLockModelMetaData &GetMetaData() const {
    return meta_;
  }

 private:
  void InitEnroll(void *model_data, size_t model_data_length) {
    enroll_sess_ = std::make_unique<Ort::Session>(
        env_, model_data, model_data_length, sess_opts_);

    GetInputNames(enroll_sess_.get(), &enroll_input_names_,
                  &enroll_input_names_ptr_);
    GetOutputNames(enroll_sess_.get(), &enroll_output_names_,
                   &enroll_output_names_ptr_);

    Ort::ModelMetadata meta_data = enroll_sess_->GetModelMetadata();
    if (config_.debug) {
      PrintInfo("voxlock enroll model", meta_data, enroll_input_names_,
                enroll_output_names_);
    }

    auto shared =
        ReadSharedMetaData(meta_data, "enroll", config_.voxlock.enroll_model);

    if (enroll_input_names_.size() != 1 || enroll_output_names_.size() != 2) {
      SHERPA_ONNX_LOGE(
          "Expect the voxlock enroll model to have 1 input and 2 outputs. "
          "Given %d inputs and %d outputs",
          static_cast<int32_t>(enroll_input_names_.size()),
          static_cast<int32_t>(enroll_output_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    meta_.version = shared.version;
    meta_.sample_rate = shared.sample_rate;
    meta_.n_fft = shared.n_fft;
    meta_.hop_length = shared.hop_length;
    meta_.window_length = shared.window_length;
    meta_.window_type = shared.window_type;

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macros below
    SHERPA_ONNX_READ_META_DATA(meta_.voice_sample_seconds,
                               "voice_sample_seconds");

    if (meta_.sample_rate <= 0 || meta_.hop_length <= 0 ||
        meta_.n_fft != meta_.window_length || meta_.n_fft % 2 != 0 ||
        meta_.n_fft <= 2 * meta_.hop_length || meta_.window_type != "hann" ||
        meta_.voice_sample_seconds <= 0) {
      SHERPA_ONNX_LOGE(
          "Unsupported voxlock meta data. sample_rate=%d, n_fft=%d, "
          "hop_length=%d, window_length=%d, window_type=%s, "
          "voice_sample_seconds=%d",
          meta_.sample_rate, meta_.n_fft, meta_.hop_length, meta_.window_length,
          meta_.window_type.c_str(), meta_.voice_sample_seconds);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitChunk(void *model_data, size_t model_data_length) {
    chunk_sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                                 model_data_length, sess_opts_);

    GetInputNames(chunk_sess_.get(), &chunk_input_names_,
                  &chunk_input_names_ptr_);
    GetOutputNames(chunk_sess_.get(), &chunk_output_names_,
                   &chunk_output_names_ptr_);

    Ort::ModelMetadata meta_data = chunk_sess_->GetModelMetadata();
    if (config_.debug) {
      PrintInfo("voxlock chunk model", meta_data, chunk_input_names_,
                chunk_output_names_);
    }

    auto shared =
        ReadSharedMetaData(meta_data, "chunk", config_.voxlock.chunk_model);

    // inputs: wav, enroll_mag, context, h, c
    // outputs: frames, h_out, c_out
    if (chunk_input_names_.size() != 5 || chunk_output_names_.size() != 3) {
      SHERPA_ONNX_LOGE(
          "Expect the voxlock chunk model to have 5 inputs and 3 outputs. "
          "Given %d inputs and %d outputs",
          static_cast<int32_t>(chunk_input_names_.size()),
          static_cast<int32_t>(chunk_output_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (shared.version != meta_.version ||
        shared.sample_rate != meta_.sample_rate ||
        shared.n_fft != meta_.n_fft || shared.hop_length != meta_.hop_length ||
        shared.window_length != meta_.window_length ||
        shared.window_type != meta_.window_type) {
      SHERPA_ONNX_LOGE(
          "The voxlock enroll and chunk models do not match.\n"
          "  enroll: version=%d, sample_rate=%d, n_fft=%d, hop_length=%d, "
          "window_length=%d, window_type=%s\n"
          "  chunk: version=%d, sample_rate=%d, n_fft=%d, hop_length=%d, "
          "window_length=%d, window_type=%s",
          meta_.version, meta_.sample_rate, meta_.n_fft, meta_.hop_length,
          meta_.window_length, meta_.window_type.c_str(), shared.version,
          shared.sample_rate, shared.n_fft, shared.hop_length,
          shared.window_length, shared.window_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macros below
    int32_t voice_sample_seconds = 0;
    SHERPA_ONNX_READ_META_DATA(voice_sample_seconds, "voice_sample_seconds");
    if (voice_sample_seconds != meta_.voice_sample_seconds) {
      SHERPA_ONNX_LOGE(
          "The voxlock enroll and chunk models do not match. "
          "voice_sample_seconds: %d vs %d",
          meta_.voice_sample_seconds, voice_sample_seconds);
      SHERPA_ONNX_EXIT(-1);
    }

    SHERPA_ONNX_READ_META_DATA(meta_.frames_per_chunk, "frames_per_chunk");
    SHERPA_ONNX_READ_META_DATA_VEC(meta_.state_shape, "state_shape");

    std::string output_gain;
    SHERPA_ONNX_READ_META_DATA_STR(output_gain, "output_gain");
    meta_.output_gain = ToFloatOrDefault(output_gain, NAN);

    if (meta_.frames_per_chunk <= 0 || !std::isfinite(meta_.output_gain)) {
      SHERPA_ONNX_LOGE(
          "Invalid voxlock meta data. frames_per_chunk=%d, output_gain=%s",
          meta_.frames_per_chunk, output_gain.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    auto h_shape =
        chunk_sess_->GetInputTypeInfo(3).GetTensorTypeAndShapeInfo().GetShape();
    auto c_shape =
        chunk_sess_->GetInputTypeInfo(4).GetTensorTypeAndShapeInfo().GetShape();
    if (h_shape != meta_.state_shape || c_shape != meta_.state_shape) {
      SHERPA_ONNX_LOGE(
          "The state shape in the voxlock chunk model does not match its meta "
          "data");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void PrintInfo(const std::string &name, const Ort::ModelMetadata &meta_data,
                 const std::vector<std::string> &input_names,
                 const std::vector<std::string> &output_names) const {
    std::ostringstream os;
    os << "---" << name << "---\n";
    PrintModelMetadata(os, meta_data);

    os << "----------input names----------\n";
    int32_t i = 0;
    for (const auto &s : input_names) {
      os << i << " " << s << "\n";
      ++i;
    }

    os << "----------output names----------\n";
    i = 0;
    for (const auto &s : output_names) {
      os << i << " " << s << "\n";
      ++i;
    }

#if __OHOS__
    SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
    SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
  }

 private:
  OfflineSpeechDenoiserModelConfig config_;
  OfflineSpeechDenoiserVoxLockModelMetaData meta_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> enroll_sess_;

  std::vector<std::string> enroll_input_names_;
  std::vector<const char *> enroll_input_names_ptr_;

  std::vector<std::string> enroll_output_names_;
  std::vector<const char *> enroll_output_names_ptr_;

  std::unique_ptr<Ort::Session> chunk_sess_;

  std::vector<std::string> chunk_input_names_;
  std::vector<const char *> chunk_input_names_ptr_;

  std::vector<std::string> chunk_output_names_;
  std::vector<const char *> chunk_output_names_ptr_;
};

OfflineSpeechDenoiserVoxLockModel::~OfflineSpeechDenoiserVoxLockModel() =
    default;  // NOLINT

OfflineSpeechDenoiserVoxLockModel::OfflineSpeechDenoiserVoxLockModel(
    const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineSpeechDenoiserVoxLockModel::OfflineSpeechDenoiserVoxLockModel(
    Manager *mgr, const OfflineSpeechDenoiserModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::pair<Ort::Value, Ort::Value> OfflineSpeechDenoiserVoxLockModel::Enroll(
    const float *samples, int32_t n) const {
  return impl_->Enroll(samples, n);
}

OfflineSpeechDenoiserVoxLockModel::States
OfflineSpeechDenoiserVoxLockModel::GetInitStates() const {
  return impl_->GetInitStates();
}

std::pair<Ort::Value, OfflineSpeechDenoiserVoxLockModel::States>
OfflineSpeechDenoiserVoxLockModel::Run(Ort::Value wav, Ort::Value enroll_mag,
                                       Ort::Value context,
                                       States states) const {
  return impl_->Run(std::move(wav), std::move(enroll_mag), std::move(context),
                    std::move(states));
}

const OfflineSpeechDenoiserVoxLockModelMetaData &
OfflineSpeechDenoiserVoxLockModel::GetMetaData() const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineSpeechDenoiserVoxLockModel::OfflineSpeechDenoiserVoxLockModel(
    AAssetManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

#if __OHOS__
template OfflineSpeechDenoiserVoxLockModel::OfflineSpeechDenoiserVoxLockModel(
    NativeResourceManager *mgr, const OfflineSpeechDenoiserModelConfig &config);
#endif

}  // namespace sherpa_onnx
