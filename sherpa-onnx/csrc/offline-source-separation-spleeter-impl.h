// sherpa-onnx/csrc/offline-source-separation-spleeter-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "kaldi-native-fbank/csrc/istft.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-source-separation-spleeter-model.h"
#include "sherpa-onnx/csrc/offline-source-separation.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

class OfflineSourceSeparationSpleeterImpl : public OfflineSourceSeparationImpl {
 public:
  explicit OfflineSourceSeparationSpleeterImpl(
      const OfflineSourceSeparationConfig &config)
      : config_(config), model_(config_.model) {}

  template <typename Manager>
  OfflineSourceSeparationSpleeterImpl(
      Manager *mgr, const OfflineSourceSeparationConfig &config)
      : config_(config), model_(mgr, config_.model) {}

  OfflineSourceSeparationOutput Process(
      const OfflineSourceSeparationInput &_input) const override {
    auto input = Resample(_input, config_.model.debug);

    auto stft_ch0 = ComputeStft(input, 0);

    auto stft_ch1 = ComputeStft(input, 1);
    knf::StftResult *p_stft_ch1 = stft_ch1.real.empty() ? &stft_ch0 : &stft_ch1;

    int32_t num_frames = stft_ch0.num_frames;
    int32_t fft_bins = stft_ch0.real.size() / num_frames;

    int32_t pad = 512 - (stft_ch0.num_frames % 512);
    if (pad < 512) {
      num_frames += pad;
    }

    if (num_frames % 512) {
      SHERPA_ONNX_LOGE("num_frames should be multiple of 512, actual: %d. %d",
                       num_frames, num_frames % 512);
      SHERPA_ONNX_EXIT(-1);
    }

    Eigen::VectorXf real(2 * num_frames * 1024);
    Eigen::VectorXf imag(2 * num_frames * 1024);
    real.setZero();
    imag.setZero();

    float *p_real = &real[0];
    float *p_imag = &imag[0];

    // copy stft result of channel 0
    for (int32_t i = 0; i != stft_ch0.num_frames; ++i) {
      std::copy(stft_ch0.real.data() + i * fft_bins,
                stft_ch0.real.data() + i * fft_bins + 1024, p_real + 1024 * i);

      std::copy(stft_ch0.imag.data() + i * fft_bins,
                stft_ch0.imag.data() + i * fft_bins + 1024, p_imag + 1024 * i);
    }

    p_real += num_frames * 1024;
    p_imag += num_frames * 1024;

    // copy stft result of channel 1
    for (int32_t i = 0; i != stft_ch1.num_frames; ++i) {
      std::copy(p_stft_ch1->real.data() + i * fft_bins,
                p_stft_ch1->real.data() + i * fft_bins + 1024,
                p_real + 1024 * i);

      std::copy(p_stft_ch1->imag.data() + i * fft_bins,
                p_stft_ch1->imag.data() + i * fft_bins + 1024,
                p_imag + 1024 * i);
    }

    Eigen::VectorXf x = (real.array().square() + imag.array().square()).sqrt();

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> x_shape{2, num_frames / 512, 512, 1024};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, &x[0], x.size(), x_shape.data(), x_shape.size());

    Ort::Value vocals_spec_tensor = model_.RunVocals(View(&x_tensor));
    Ort::Value accompaniment_spec_tensor =
        model_.RunAccompaniment(std::move(x_tensor));

    Eigen::VectorXf vocals_spec = Eigen::Map<Eigen::VectorXf>(
        vocals_spec_tensor.GetTensorMutableData<float>(), x.size());

    Eigen::VectorXf accompaniment_spec = Eigen::Map<Eigen::VectorXf>(
        accompaniment_spec_tensor.GetTensorMutableData<float>(), x.size());

    Eigen::VectorXf sum_spec = vocals_spec.array().square() +
                               accompaniment_spec.array().square() + 1e-10;

    vocals_spec = (vocals_spec.array().square() + 1e-10 / 2) / sum_spec.array();

    accompaniment_spec =
        (accompaniment_spec.array().square() + 1e-10 / 2) / sum_spec.array();

    auto vocals_samples_ch0 = ProcessSpec(vocals_spec, stft_ch0, 0);
    auto vocals_samples_ch1 = ProcessSpec(vocals_spec, *p_stft_ch1, 1);

    auto accompaniment_samples_ch0 =
        ProcessSpec(accompaniment_spec, stft_ch0, 0);
    auto accompaniment_samples_ch1 =
        ProcessSpec(accompaniment_spec, *p_stft_ch1, 1);

    OfflineSourceSeparationOutput ans;
    ans.sample_rate = GetOutputSampleRate();

    ans.stems.resize(2);
    ans.stems[0].data.reserve(2);
    ans.stems[1].data.reserve(2);

    ans.stems[0].data.push_back(std::move(vocals_samples_ch0));
    ans.stems[0].data.push_back(std::move(vocals_samples_ch1));

    ans.stems[1].data.push_back(std::move(accompaniment_samples_ch0));
    ans.stems[1].data.push_back(std::move(accompaniment_samples_ch1));

    return ans;
  }

  int32_t GetOutputSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

  int32_t GetNumberOfStems() const override {
    return model_.GetMetaData().num_stems;
  }

 private:
  // spec is of shape (2, num_chunks, 512, 1024)
  std::vector<float> ProcessSpec(const Eigen::VectorXf &spec,
                                 const knf::StftResult &stft,
                                 int32_t channel) const {
    int32_t fft_bins = stft.real.size() / stft.num_frames;

    Eigen::VectorXf mask(stft.real.size());
    mask.setZero();

    float *p_mask = &mask[0];

    // assume there are 2 channels
    const float *p_spec = &spec[0] + (spec.size() / 2) * channel;

    for (int32_t i = 0; i != stft.num_frames; ++i) {
      std::copy(p_spec + i * 1024, p_spec + (i + 1) * 1024,
                p_mask + i * fft_bins);
    }

    knf::StftResult masked_stft;

    masked_stft.num_frames = stft.num_frames;
    masked_stft.real.resize(stft.real.size());
    masked_stft.imag.resize(stft.imag.size());

    Eigen::Map<Eigen::VectorXf>(masked_stft.real.data(),
                                masked_stft.real.size()) =
        mask.array() *
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(stft.real.data()),
                                    stft.real.size())
            .array();

    Eigen::Map<Eigen::VectorXf>(masked_stft.imag.data(),
                                masked_stft.imag.size()) =
        mask.array() *
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(stft.imag.data()),
                                    stft.imag.size())
            .array();

    auto stft_config = GetStftConfig();
    knf::IStft istft(stft_config);

    return istft.Compute(masked_stft);
  }

  knf::StftResult ComputeStft(const OfflineSourceSeparationInput &input,
                              int32_t ch) const {
    if (ch >= input.samples.data.size()) {
      SHERPA_ONNX_LOGE("Invalid channel %d. Max %d", ch,
                       static_cast<int32_t>(input.samples.data.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (input.samples.data[ch].empty()) {
      return {};
    }

    return ComputeStft(input.samples.data[ch]);
  }

  knf::StftResult ComputeStft(const std::vector<float> &samples) const {
    auto stft_config = GetStftConfig();
    knf::Stft stft(stft_config);

    return stft.Compute(samples.data(), samples.size());
  }

  knf::StftConfig GetStftConfig() const {
    const auto &meta = model_.GetMetaData();

    knf::StftConfig stft_config;
    stft_config.n_fft = meta.n_fft;
    stft_config.hop_length = meta.hop_length;
    stft_config.win_length = meta.window_length;
    stft_config.window_type = meta.window_type;
    stft_config.center = meta.center;

    return stft_config;
  }

 private:
  OfflineSourceSeparationConfig config_;
  OfflineSourceSeparationSpleeterModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_SPLEETER_IMPL_H_
