// sherpa-onnx/csrc/offline-tts-zipvoice-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-zipvoice-model.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
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

class OfflineTtsZipvoiceModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto text_buf = ReadFile(config.zipvoice.text_model);
    auto fm_buf = ReadFile(config.zipvoice.flow_matching_model);
    Init(text_buf.data(), text_buf.size(), fm_buf.data(), fm_buf.size());
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto text_buf = ReadFile(mgr, config.zipvoice.text_model);
    auto fm_buf = ReadFile(mgr, config.zipvoice.flow_matching_model);
    Init(text_buf.data(), text_buf.size(), fm_buf.data(), fm_buf.size());
  }

  const OfflineTtsZipvoiceModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  Ort::Value Run(Ort::Value tokens, Ort::Value prompt_tokens,
                 Ort::Value prompt_features, float speed, int32_t num_steps) {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<int64_t> tokens_shape =
        tokens.GetTensorTypeAndShapeInfo().GetShape();
    int64_t batch_size = tokens_shape[0];
    if (batch_size != 1) {
      SHERPA_ONNX_LOGE("Support only batch_size == 1. Given: %d",
                       static_cast<int32_t>(batch_size));
      SHERPA_ONNX_EXIT(-1);
    }

    std::vector<int64_t> prompt_feat_shape =
        prompt_features.GetTensorTypeAndShapeInfo().GetShape();

    int64_t prompt_feat_len = prompt_feat_shape[1];
    int64_t prompt_feat_len_shape = 1;
    Ort::Value prompt_feat_len_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, &prompt_feat_len, 1, &prompt_feat_len_shape, 1);

    int64_t speed_shape = 1;
    Ort::Value speed_tensor = Ort::Value::CreateTensor<float>(
        memory_info, &speed, 1, &speed_shape, 1);

    std::vector<Ort::Value> text_inputs;
    text_inputs.reserve(4);
    text_inputs.push_back(std::move(tokens));
    text_inputs.push_back(std::move(prompt_tokens));
    text_inputs.push_back(std::move(prompt_feat_len_tensor));
    text_inputs.push_back(std::move(speed_tensor));

    // forward text-encoder
    auto text_out =
        text_sess_->Run({}, text_input_names_ptr_.data(), text_inputs.data(),
                        text_inputs.size(), text_output_names_ptr_.data(),
                        text_output_names_ptr_.size());

    Ort::Value &text_condition = text_out[0];

    std::vector<int64_t> text_cond_shape =
        text_condition.GetTensorTypeAndShapeInfo().GetShape();
    int64_t num_frames = text_cond_shape[1];

    int64_t feat_dim = meta_data_.feat_dim;

    std::vector<float> x_data(batch_size * num_frames * feat_dim);
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::normal_distribution<float> norm(0, 1);
    for (auto &v : x_data) {
      v = norm(rng);
    }

    std::vector<int64_t> x_shape = {batch_size, num_frames, feat_dim};
    Ort::Value x = Ort::Value::CreateTensor<float>(
        memory_info, x_data.data(), x_data.size(), x_shape.data(),
        x_shape.size());

    std::vector<float> speech_cond_data(batch_size * num_frames * feat_dim,
                                        0.0f);
    const float *src = prompt_features.GetTensorData<float>();
    float *dst = speech_cond_data.data();
    std::memcpy(dst, src,
                batch_size * prompt_feat_len * feat_dim * sizeof(float));
    std::vector<int64_t> speech_cond_shape = {batch_size, num_frames, feat_dim};
    Ort::Value speech_condition = Ort::Value::CreateTensor<float>(
        memory_info, speech_cond_data.data(), speech_cond_data.size(),
        speech_cond_shape.data(), speech_cond_shape.size());

    float t_shift = config_.zipvoice.t_shift;
    float guidance_scale = config_.zipvoice.guidance_scale;

    std::vector<float> timesteps(num_steps + 1);
    for (int32_t i = 0; i <= num_steps; ++i) {
      float t = static_cast<float>(i) / num_steps;
      timesteps[i] = t_shift * t / (1.0f + (t_shift - 1.0f) * t);
    }

    int64_t guidance_scale_shape = 1;
    Ort::Value guidance_scale_tensor = Ort::Value::CreateTensor<float>(
        memory_info, &guidance_scale, 1, &guidance_scale_shape, 1);

    std::vector<Ort::Value> fm_inputs;
    fm_inputs.reserve(5);
    // fm_inputs[0] is t tensor, will set in for loop
    fm_inputs.emplace_back(nullptr);
    fm_inputs.push_back(std::move(x));
    fm_inputs.push_back(std::move(text_condition));
    fm_inputs.push_back(std::move(speech_condition));
    fm_inputs.push_back(std::move(guidance_scale_tensor));

    for (int32_t step = 0; step < num_steps; ++step) {
      float t_val = timesteps[step];
      int64_t t_shape = 1;
      Ort::Value t_tensor =
          Ort::Value::CreateTensor<float>(memory_info, &t_val, 1, &t_shape, 1);
      fm_inputs[0] = std::move(t_tensor);
      auto fm_out = fm_sess_->Run(
          {}, fm_input_names_ptr_.data(), fm_inputs.data(), fm_inputs.size(),
          fm_output_names_ptr_.data(), fm_output_names_ptr_.size());
      Ort::Value &v = fm_out[0];

      float delta_t = timesteps[step + 1] - timesteps[step];
      float *x_ptr = fm_inputs[1].GetTensorMutableData<float>();
      const float *v_ptr = v.GetTensorData<float>();
      int64_t N = batch_size * num_frames * feat_dim;
      for (int64_t i = 0; i < N; ++i) {
        x_ptr[i] += v_ptr[i] * delta_t;
      }
    }

    int64_t keep_frames = num_frames - prompt_feat_len;
    std::vector<float> out_data(batch_size * keep_frames * feat_dim);
    x = std::move(fm_inputs[1]);
    const float *x_ptr = x.GetTensorData<float>();
    for (int64_t b = 0; b < batch_size; ++b) {
      std::memcpy(out_data.data() + b * keep_frames * feat_dim,
                  x_ptr + (b * num_frames + prompt_feat_len) * feat_dim,
                  keep_frames * feat_dim * sizeof(float));
    }
    std::vector<int64_t> out_shape = {batch_size, keep_frames, feat_dim};

    Ort::Value ans = Ort::Value::CreateTensor<float>(
        allocator_, out_shape.data(), out_shape.size());

    std::copy(out_data.begin(), out_data.end(),
              ans.GetTensorMutableData<float>());

    return ans;
  }

 private:
  void Init(void *text_model_data, size_t text_model_data_length,
            void *fm_model_data, size_t fm_model_data_length) {
    // Init text-encoder model
    text_sess_ = std::make_unique<Ort::Session>(
        env_, text_model_data, text_model_data_length, sess_opts_);
    GetInputNames(text_sess_.get(), &text_input_names_, &text_input_names_ptr_);
    GetOutputNames(text_sess_.get(), &text_output_names_,
                   &text_output_names_ptr_);

    // Init flow-matching model
    fm_sess_ = std::make_unique<Ort::Session>(env_, fm_model_data,
                                              fm_model_data_length, sess_opts_);
    GetInputNames(fm_sess_.get(), &fm_input_names_, &fm_input_names_ptr_);
    GetOutputNames(fm_sess_.get(), &fm_output_names_, &fm_output_names_ptr_);

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    Ort::ModelMetadata meta_data = text_sess_->GetModelMetadata();
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.use_espeak, "use_espeak",
                                            1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.use_pinyin, "use_pinyin",
                                            1);

    meta_data = fm_sess_->GetModelMetadata();
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.feat_dim, "feat_dim",
                                            100);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.sample_rate,
                                            "sample_rate", 24000);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.n_fft, "n_fft", 1024);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.hop_length, "hop_length",
                                            256);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.window_length,
                                            "window_length", 1024);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.num_mels, "num_mels",
                                            100);

    if (config_.debug) {
      std::ostringstream os;

      os << "---zipvoice text-encoder model---\n";
      Ort::ModelMetadata text_meta_data = text_sess_->GetModelMetadata();
      PrintModelMetadata(os, text_meta_data);

      os << "----------input names----------\n";
      int32_t i = 0;
      for (const auto &s : text_input_names_) {
        os << i << " " << s << "\n";
        ++i;
      }
      os << "----------output names----------\n";
      i = 0;
      for (const auto &s : text_output_names_) {
        os << i << " " << s << "\n";
        ++i;
      }

      os << "---zipvoice flow-matching model---\n";
      PrintModelMetadata(os, meta_data);

      os << "----------input names----------\n";
      i = 0;
      for (const auto &s : fm_input_names_) {
        os << i << " " << s << "\n";
        ++i;
      }
      os << "----------output names----------\n";
      i = 0;
      for (const auto &s : fm_output_names_) {
        os << i << " " << s << "\n";
        ++i;
      }

#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }
  }

 private:
  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> text_sess_;
  std::unique_ptr<Ort::Session> fm_sess_;

  std::vector<std::string> text_input_names_;
  std::vector<const char *> text_input_names_ptr_;

  std::vector<std::string> text_output_names_;
  std::vector<const char *> text_output_names_ptr_;

  std::vector<std::string> fm_input_names_;
  std::vector<const char *> fm_input_names_ptr_;

  std::vector<std::string> fm_output_names_;
  std::vector<const char *> fm_output_names_ptr_;

  OfflineTtsZipvoiceModelMetaData meta_data_;
};

OfflineTtsZipvoiceModel::OfflineTtsZipvoiceModel(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsZipvoiceModel::OfflineTtsZipvoiceModel(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineTtsZipvoiceModel::~OfflineTtsZipvoiceModel() = default;

const OfflineTtsZipvoiceModelMetaData &OfflineTtsZipvoiceModel::GetMetaData()
    const {
  return impl_->GetMetaData();
}

Ort::Value OfflineTtsZipvoiceModel::Run(Ort::Value tokens,
                                        Ort::Value prompt_tokens,
                                        Ort::Value prompt_features,
                                        float speed /*= 1.0*/,
                                        int32_t num_steps /*= 16*/) const {
  return impl_->Run(std::move(tokens), std::move(prompt_tokens),
                    std::move(prompt_features), speed, num_steps);
}

#if __ANDROID_API__ >= 9
template OfflineTtsZipvoiceModel::OfflineTtsZipvoiceModel(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsZipvoiceModel::OfflineTtsZipvoiceModel(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
