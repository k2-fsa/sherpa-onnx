// sherpa-onnx/csrc/vocos-vocoder.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/vocos-vocoder.h"

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

#include "kaldi-native-fbank/csrc/istft.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

struct VocosModelMetaData {
  int32_t n_fft;
  int32_t hop_length;
  int32_t win_length;
  int32_t center;
  int32_t normalized;
  std::string window_type;
  std::string pad_mode;
};

class VocosVocoder::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config.num_threads, config.provider)),
        allocator_{} {
    auto buf = ReadFile(config.matcha.vocoder);
    Init(buf.data(), buf.size());
  }

  template <typename Manager>
  explicit Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config.num_threads, config.provider)),
        allocator_{} {
    auto buf = ReadFile(mgr, config.matcha.vocoder);
    Init(buf.data(), buf.size());
  }

  std::vector<float> Run(Ort::Value mel) const {
    auto out = sess_->Run({}, input_names_ptr_.data(), &mel, 1,
                          output_names_ptr_.data(), output_names_ptr_.size());

    std::vector<int64_t> shape = out[0].GetTensorTypeAndShapeInfo().GetShape();

    if (shape[0] != 1) {
      SHERPA_ONNX_LOGE("Support only batch size 1, given: %d",
                       static_cast<int32_t>(shape[0]));
      SHERPA_ONNX_EXIT(-1);
    }

    knf::StftResult stft_result;
    stft_result.num_frames = shape[2];
    stft_result.real.resize(shape[1] * shape[2]);
    stft_result.imag.resize(shape[1] * shape[2]);

    // stft_result.real: (num_frames, n_fft/2+1), flattened in row major

    // mag.shape: (batch_size, n_fft/2+1, num_frames)
    const float *p_mag = out[0].GetTensorData<float>();
    const float *p_x = out[1].GetTensorData<float>();
    const float *p_y = out[2].GetTensorData<float>();

    for (int32_t frame_index = 0; frame_index < static_cast<int32_t>(shape[2]);
         ++frame_index) {
      for (int32_t bin = 0; bin < static_cast<int32_t>(shape[1]); ++bin) {
        stft_result.real[frame_index * shape[1] + bin] =
            p_mag[bin * shape[2] + frame_index] *
            p_x[bin * shape[2] + frame_index];
        stft_result.imag[frame_index * shape[1] + bin] =
            p_mag[bin * shape[2] + frame_index] *
            p_y[bin * shape[2] + frame_index];
      }
    }

    knf::StftConfig stft_config;
    stft_config.n_fft = meta_.n_fft;
    stft_config.hop_length = meta_.hop_length;
    stft_config.win_length = meta_.win_length;
    stft_config.normalized = meta_.normalized;
    stft_config.center = meta_.center;
    stft_config.window_type = meta_.window_type;
    stft_config.pad_mode = meta_.pad_mode;

    knf::IStft istft(stft_config);
    return istft.Compute(stft_result);
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      os << "---Vocos model---\n";
      PrintModelMetadata(os, meta_data);

      os << "----------input names----------\n";
      int32_t i = 0;
      for (const auto &s : input_names_) {
        os << i << " " << s << "\n";
        ++i;
      }
      os << "----------output names----------\n";
      i = 0;
      for (const auto &s : output_names_) {
        os << i << " " << s << "\n";
        ++i;
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_.n_fft, "n_fft");
    SHERPA_ONNX_READ_META_DATA(meta_.hop_length, "hop_length");
    SHERPA_ONNX_READ_META_DATA(meta_.win_length, "win_length");
    SHERPA_ONNX_READ_META_DATA(meta_.center, "center");
    SHERPA_ONNX_READ_META_DATA(meta_.normalized, "normalized");
    SHERPA_ONNX_READ_META_DATA_STR(meta_.window_type, "window_type");
    SHERPA_ONNX_READ_META_DATA_STR(meta_.pad_mode, "pad_mode");
  }

 private:
  OfflineTtsModelConfig config_;
  VocosModelMetaData meta_;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
};

VocosVocoder::VocosVocoder(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
VocosVocoder::VocosVocoder(Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

VocosVocoder::~VocosVocoder() = default;

std::vector<float> VocosVocoder::Run(Ort::Value mel) const {
  return impl_->Run(std::move(mel));
}

#if __ANDROID_API__ >= 9
template VocosVocoder::VocosVocoder(AAssetManager *mgr,
                                    const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template VocosVocoder::VocosVocoder(NativeResourceManager *mgr,
                                    const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
