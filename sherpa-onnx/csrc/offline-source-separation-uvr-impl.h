// sherpa-onnx/csrc/offline-source-separation-uvr-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_IMPL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "kaldi-native-fbank/csrc/istft.h"
#include "kaldi-native-fbank/csrc/stft.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-source-separation-uvr-model.h"
#include "sherpa-onnx/csrc/offline-source-separation.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/resample.h"

namespace sherpa_onnx {

class OfflineSourceSeparationUvrImpl : public OfflineSourceSeparationImpl {
 public:
  explicit OfflineSourceSeparationUvrImpl(
      const OfflineSourceSeparationConfig &config)
      : config_(config), model_(config_.model) {}

  template <typename Manager>
  OfflineSourceSeparationUvrImpl(Manager *mgr,
                                 const OfflineSourceSeparationConfig &config)
      : config_(config), model_(mgr, config_.model) {}

  OfflineSourceSeparationOutput Process(
      const OfflineSourceSeparationInput &_input) const override {
    auto input = Resample(_input, config_.model.debug);

    auto chunks_ch0 = SplitIntoChunks(input.samples.data[0]);

    std::vector<std::vector<float>> chunks_ch1;
    if (input.samples.data.size() > 1) {
      chunks_ch1 = SplitIntoChunks(input.samples.data[1]);
    }

    std::vector<float> samples_ch0;
    std::vector<float> samples_ch1;

    for (int32_t i = 0; i != static_cast<int32_t>(chunks_ch0.size()); ++i) {
      bool is_first_chunk = (i == 0);
      bool is_last_chunk = (i == static_cast<int32_t>(chunks_ch0.size()) - 1);

      auto s = ProcessChunk(
          chunks_ch0[i],
          chunks_ch1.empty() ? std::vector<float>{} : chunks_ch1[i],
          is_first_chunk, is_last_chunk);

      samples_ch0.insert(samples_ch0.end(), s.first.begin(), s.first.end());
      samples_ch1.insert(samples_ch1.end(), s.second.begin(), s.second.end());
    }

    auto &vocals_ch0 = samples_ch0;
    auto &vocals_ch1 = samples_ch1;

    std::vector<float> non_vocals_ch0(vocals_ch0.size());
    std::vector<float> non_vocals_ch1(vocals_ch1.size());

    Eigen::Map<Eigen::VectorXf>(non_vocals_ch0.data(), non_vocals_ch0.size()) =
        Eigen::Map<Eigen::VectorXf>(input.samples.data[0].data(),
                                    input.samples.data[0].size())
            .array() -
        Eigen::Map<Eigen::VectorXf>(vocals_ch0.data(), vocals_ch0.size())
            .array();

    if (input.samples.data.size() > 1) {
      Eigen::Map<Eigen::VectorXf>(non_vocals_ch1.data(),
                                  non_vocals_ch1.size()) =
          Eigen::Map<Eigen::VectorXf>(input.samples.data[1].data(),
                                      input.samples.data[1].size())
              .array() -
          Eigen::Map<Eigen::VectorXf>(vocals_ch1.data(), vocals_ch1.size())
              .array();
    } else {
      Eigen::Map<Eigen::VectorXf>(non_vocals_ch1.data(),
                                  non_vocals_ch1.size()) =
          Eigen::Map<Eigen::VectorXf>(input.samples.data[0].data(),
                                      input.samples.data[0].size())
              .array() -
          Eigen::Map<Eigen::VectorXf>(vocals_ch1.data(), vocals_ch1.size())
              .array();
    }

    OfflineSourceSeparationOutput ans;
    ans.sample_rate = GetOutputSampleRate();

    ans.stems.resize(2);
    ans.stems[0].data.reserve(2);
    ans.stems[1].data.reserve(2);

    ans.stems[0].data.push_back(std::move(vocals_ch0));
    ans.stems[0].data.push_back(std::move(vocals_ch1));

    ans.stems[1].data.push_back(std::move(non_vocals_ch0));
    ans.stems[1].data.push_back(std::move(non_vocals_ch1));

    return ans;
  }

  int32_t GetOutputSampleRate() const override {
    return model_.GetMetaData().sample_rate;
  }

  int32_t GetNumberOfStems() const override {
    return model_.GetMetaData().num_stems;
  }

 private:
  std::pair<std::vector<float>, std::vector<float>> ProcessChunk(
      const std::vector<float> &chunk_ch0, const std::vector<float> &chunk_ch1,
      bool is_first_chunk, bool is_last_chunk) const {
    int32_t pad0 = 0;

    auto stft_results_ch0 = ComputeStft(chunk_ch0, &pad0);

    int32_t pad1 = pad0;
    std::vector<knf::StftResult> stft_results_ch1;

    if (!chunk_ch1.empty()) {
      stft_results_ch1 = ComputeStft(chunk_ch1, &pad1);
    } else {
      stft_results_ch1 = stft_results_ch0;
    }

    const auto &meta_ = model_.GetMetaData();

    int32_t num_frames = stft_results_ch0[0].num_frames;
    int32_t dim_f = meta_.dim_f;
    int32_t dim_t = meta_.dim_t;
    int32_t n_fft_bin = meta_.n_fft / 2 + 1;
    if (num_frames != dim_t) {
      SHERPA_ONNX_LOGE("num_frames(%d) != dim_t(%d)", num_frames, dim_t);
      SHERPA_ONNX_EXIT(-1);
    }

    // the first 2: number of channels
    // the second 2: real and image
    std::vector<float> x(stft_results_ch0.size() * 2 * 2 * dim_f * dim_t);
    float *px = x.data();

    for (int32_t i = 0; i != static_cast<int32_t>(stft_results_ch0.size());
         ++i) {
      const auto &ch0 = stft_results_ch0[i];
      const auto &ch1 = stft_results_ch1[i];

      const float *p_real_ch0 = ch0.real.data();
      const float *p_imag_ch0 = ch0.imag.data();

      const float *p_real_ch1 = ch1.real.data();
      const float *p_imag_ch1 = ch1.imag.data();

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          *px = p_real_ch0[k * n_fft_bin + j];
          ++px;
        }
      }

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          *px = p_imag_ch0[k * n_fft_bin + j];
          ++px;
        }
      }

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          *px = p_real_ch1[k * n_fft_bin + j];
          ++px;
        }
      }

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          *px = p_imag_ch1[k * n_fft_bin + j];
          ++px;
        }
      }
    }  // for (int32_t i = 0; i !=

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> x_shape{
        static_cast<int32_t>(stft_results_ch0.size()) * 4 / meta_.dim_c,
        meta_.dim_c, dim_f, dim_t};

    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    Ort::Value spec = model_.Run(std::move(x_tensor));

    const float *p_spec = spec.GetTensorData<float>();

    for (int32_t i = 0; i != static_cast<int32_t>(stft_results_ch0.size());
         ++i) {
      auto &ch0 = stft_results_ch0[i];
      auto &ch1 = stft_results_ch1[i];

      float *p_real_ch0 = ch0.real.data();
      float *p_imag_ch0 = ch0.imag.data();

      float *p_real_ch1 = ch1.real.data();
      float *p_imag_ch1 = ch1.imag.data();

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          p_real_ch0[k * n_fft_bin + j] = *p_spec;
          ++p_spec;
        }
      }

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          p_imag_ch0[k * n_fft_bin + j] = *p_spec;
          ++p_spec;
        }
      }

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          p_real_ch1[k * n_fft_bin + j] = *p_spec;
          ++p_spec;
        }
      }

      for (int32_t j = 0; j != dim_f; ++j) {
        for (int32_t k = 0; k != num_frames; ++k) {
          p_imag_ch1[k * n_fft_bin + j] = *p_spec;
          ++p_spec;
        }
      }

      for (int32_t k = 0; k != num_frames; ++k) {
        for (int32_t j = dim_f; j != n_fft_bin; ++j) {
          p_real_ch0[k * n_fft_bin + j] = 0;
          p_real_ch1[k * n_fft_bin + j] = 0;

          p_imag_ch0[k * n_fft_bin + j] = 0;
          p_imag_ch1[k * n_fft_bin + j] = 0;
        }
      }
    }

    auto samples_ch0 = ComputeInverseStft(stft_results_ch0, pad0,
                                          is_first_chunk, is_last_chunk);

    auto samples_ch1 = ComputeInverseStft(stft_results_ch1, pad1,
                                          is_first_chunk, is_last_chunk);

    return {std::move(samples_ch0), std::move(samples_ch1)};
  }

  std::vector<float> ComputeInverseStft(
      const std::vector<knf::StftResult> &stft_result, int32_t pad,
      bool is_first_chunk, bool is_last_chunk) const {
    const auto &meta_ = model_.GetMetaData();
    int32_t trim = meta_.n_fft / 2;

    int32_t margin = meta_.margin;

    int32_t chunk_size = meta_.num_chunks * meta_.sample_rate;

    if (margin > chunk_size) {
      margin = chunk_size;
    }

    auto stft_config = GetStftConfig();
    knf::IStft istft(stft_config);

    std::vector<float> ans;

    for (int32_t i = 0; i != static_cast<int32_t>(stft_result.size()); ++i) {
      auto samples = istft.Compute(stft_result[i]);
      int32_t num_samples = static_cast<int32_t>(samples.size());

      ans.insert(ans.end(), samples.begin() + trim,
                 samples.begin() + (num_samples - trim));
    }

    int32_t start = is_first_chunk ? 0 : margin;
    int32_t end =
        is_last_chunk ? (ans.size() - pad) : (ans.size() - pad - margin);

    return {ans.begin() + start, ans.begin() + end};
  }

  std::vector<knf::StftResult> ComputeStft(const std::vector<float> &chunk,
                                           int32_t *pad) const {
    const auto &meta_ = model_.GetMetaData();

    int32_t num_samples = static_cast<int32_t>(chunk.size());
    int32_t trim = meta_.n_fft / 2;
    int32_t chunk_size = meta_.hop_length * (meta_.dim_t - 1);
    int32_t gen_size = chunk_size - 2 * trim;
    *pad = gen_size - num_samples % gen_size;

    std::vector<float> samples(trim + chunk.size() + *pad + trim);
    std::copy(chunk.begin(), chunk.end(), samples.begin() + trim);

    auto stft_config = GetStftConfig();
    knf::Stft stft(stft_config);

    std::vector<knf::StftResult> stft_results;
    // split the chunk into short segments
    for (int32_t i = 0; i < num_samples + *pad; i += gen_size) {
      auto r = stft.Compute(samples.data() + i, chunk_size);
      stft_results.push_back(std::move(r));
    }

    return stft_results;
  }

  std::vector<std::vector<float>> SplitIntoChunks(
      const std::vector<float> &samples) const {
    std::vector<std::vector<float>> ans;

    if (samples.empty()) {
      return ans;
    }

    const auto &meta_ = model_.GetMetaData();
    int32_t margin = meta_.margin;

    int32_t chunk_size = meta_.num_chunks * meta_.sample_rate;

    if (static_cast<int32_t>(samples.size()) < chunk_size) {
      chunk_size = samples.size();
    }

    if (margin > chunk_size) {
      margin = chunk_size;
    }

    for (int32_t i = 0; i < static_cast<int32_t>(samples.size());
         i += chunk_size) {
      int32_t start = std::max<int32_t>(0, i - margin);
      int32_t end = std::min<int32_t>(i + chunk_size + margin,
                                      static_cast<int32_t>(samples.size()));
      if (start >= end) {
        break;
      }

      ans.emplace_back(samples.begin() + start, samples.begin() + end);

      if (end == static_cast<int32_t>(samples.size())) {
        break;
      }
    }

    return ans;
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
  OfflineSourceSeparationUvrModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SOURCE_SEPARATION_UVR_IMPL_H_
