// sherpa-onnx/csrc/offline-speaker-diarization-sortformer-impl.h
//
// Copyright (c)  2026  Silvio Tomatis
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_SORTFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_SORTFORMER_IMPL_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model.h"
#include "sherpa-onnx/csrc/sortformer-speaker-cache.h"
#include "sherpa-onnx/csrc/timer.h"

namespace sherpa_onnx {

// End-to-end speaker diarization with the streaming Sortformer, e.g.,
// https://huggingface.co/nvidia/Nemotron-3-Diarization
//
// The audio is processed in chunks. Each chunk attends to the
// Arrival-Order Speaker Cache and the FIFO queue of the previous chunks,
// so speakers keep their IDs across chunks and recordings of any length
// can be processed. Speaker IDs are in the order of first arrival.
class OfflineSpeakerDiarizationSortformerImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationSortformerImpl() override = default;

  explicit OfflineSpeakerDiarizationSortformerImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config), model_(config_.segmentation) {
    Init();
  }

  template <typename Manager>
  OfflineSpeakerDiarizationSortformerImpl(
      Manager *mgr, const OfflineSpeakerDiarizationConfig &config)
      : config_(config), model_(mgr, config_.segmentation) {
    Init();
  }

  int32_t SampleRate() const override {
    return model_.GetModelMetaData().sample_rate;
  }

  void SetConfig(const OfflineSpeakerDiarizationConfig & /*config*/) override {
    // The number of speakers is decided by the model. There is no
    // clustering to configure.
    if (config_.segmentation.debug) {
      SHERPA_ONNX_LOGE(
          "SetConfig() has no effect on Sortformer speaker diarization");
    }
  }

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    std::vector<float> probs =
        ComputeSpeakerProbs(audio, n, callback, callback_arg);
    return ComputeResult(probs);
  }

  // Return a row-major matrix of shape (num_frames, num_speakers) with the
  // activity probability of each speaker in each 10 ms frame.
  // num_frames is n / hop_length.
  std::vector<float> ComputeSpeakerProbs(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const {
    const auto &meta = model_.GetModelMetaData();
    const int32_t num_speakers = meta.cache.num_speakers;
    const int32_t factor = meta.cache.subsampling_factor;
    const int32_t hidden_size = meta.cache.hidden_size;

    if (!audio || n <= 0) {
      return {};
    }

    int32_t num_frames = n / meta.hop_length;
    if (num_frames == 0) {
      return {};
    }

    Timer timer(config_.segmentation.debug);

    // The centered STFT has an extra, masked feature frame. Keep its
    // zero-padded embedding: it contributes to the output convolution.
    int32_t num_embeds = num_frames / factor + 1;
    int32_t num_chunks =
        (num_embeds + meta.chunk_length - 1) / meta.chunk_length;

    std::vector<float> ans(static_cast<int64_t>(num_frames) * num_speakers);

    SortformerSpeakerCache cache(meta.cache);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    for (int32_t k = 0; k != num_chunks; ++k) {
      int32_t start = k * meta.chunk_length;
      int32_t end = std::min(start + meta.chunk_length, num_embeds);
      int32_t stop = std::min(end + meta.chunk_right_context, num_embeds);
      int32_t num_chunk_frames = end - start;
      int32_t num_step_frames = stop - start;

      std::vector<float> features = ComputeFeatures(
          audio, n, start * factor, num_step_frames * factor, num_frames);

      std::array<int64_t, 3> features_shape{1, num_step_frames * factor,
                                            meta.num_mel_bins};
      Ort::Value features_tensor = Ort::Value::CreateTensor(
          memory_info, features.data(), features.size(), features_shape.data(),
          features_shape.size());

      int32_t num_cached = cache.NumCachedFrames();
      std::array<int64_t, 3> cached_shape{1, num_cached, hidden_size};
      Ort::Value cached_tensor = Ort::Value::CreateTensor<float>(
          model_.Allocator(), cached_shape.data(), cached_shape.size());
      if (num_cached > 0) {
        std::vector<float> cached = cache.GetEmbeds();
        std::copy(cached.begin(), cached.end(),
                  cached_tensor.GetTensorMutableData<float>());
      }

      auto [probs, chunk_embeds] =
          model_.Forward(std::move(features_tensor), std::move(cached_tensor),
                         std::min(stop * factor, num_frames) - start * factor);

      const float *p = probs.GetTensorData<float>();
      int32_t num_input_frames = num_cached + num_step_frames;

      // Keep the probabilities of the chunk frames, without the cached frames
      // and the right context
      int32_t offset = start * factor;
      int32_t count = std::min(num_chunk_frames * factor, num_frames - offset);
      const float *src =
          p + static_cast<int64_t>(num_cached) * factor * num_speakers;
      std::copy(src, src + static_cast<int64_t>(count) * num_speakers,
                ans.begin() + static_cast<int64_t>(offset) * num_speakers);

      // Padding probabilities are not zeroed before pooling. Padding occurs
      // only in the final chunk, so this cache state is never used again.
      cache.Update(chunk_embeds.GetTensorData<float>(), num_chunk_frames, p,
                   num_input_frames);

      if (callback) {
        callback(k + 1, num_chunks, callback_arg);
      }
    }

    timer.Log("OfflineSpeakerDiarization: sortformer");

    return ans;
  }

 private:
  void Init() {
    const auto &meta = model_.GetModelMetaData();

    knf::FbankOptions &opts = fbank_opts_;
    opts.frame_opts.samp_freq = meta.sample_rate;
    opts.frame_opts.frame_length_ms =
        1000.0f * meta.win_length / meta.sample_rate;
    opts.frame_opts.frame_shift_ms =
        1000.0f * meta.hop_length / meta.sample_rate;
    opts.frame_opts.dither = 0;
    opts.frame_opts.remove_dc_offset = false;
    // We apply the preemphasis over the whole waveform in ComputeFeatures()
    opts.frame_opts.preemph_coeff = 0;
    // A symmetric Hann window, like torch.hann_window(periodic=False)
    opts.frame_opts.window_type = "hanning";
    opts.frame_opts.snip_edges = true;
    opts.frame_opts.round_to_power_of_two = true;
    opts.mel_opts.num_bins = meta.num_mel_bins;
    opts.mel_opts.low_freq = 0;
    opts.mel_opts.high_freq = 0;  // Nyquist
    opts.mel_opts.is_librosa = true;
    opts.use_power = true;
    opts.use_log_fbank = false;

    // kaldi-native-fbank derives the FFT size from the window length.
    // Reject an incompatible frontend rather than silently changing it.
    if (opts.frame_opts.PaddedWindowSize() != meta.n_fft ||
        opts.frame_opts.WindowSize() != meta.win_length ||
        opts.frame_opts.WindowShift() != meta.hop_length ||
        meta.win_length % 2 != 0) {
      SHERPA_ONNX_LOGE("Unsupported Sortformer frontend geometry");
      SHERPA_ONNX_EXIT(-1);
    }

    if (config_.segmentation.debug) {
      SHERPA_ONNX_LOGE("%s", opts.ToString().c_str());
    }
  }

  /* Compute num_out frames of log-mel features starting at frame start.
   *
   * It matches NeMo's AudioToMelSpectrogramPreprocessor without
   * normalization: preemphasis over the whole waveform, then
   * torch.stft(center=True, pad_mode="constant"). A win_length window centered
   * in n_fft gives the same power spectrum as a win_length window starting
   * win_length / 2 samples before the frame center.
   *
   * Frames at and after num_frames are zero, like the padding of the last
   * encoder frame in the reference implementation.
   */
  std::vector<float> ComputeFeatures(const float *audio, int32_t n,
                                     int32_t start, int32_t num_out,
                                     int32_t num_frames) const {
    const auto &meta = model_.GetModelMetaData();
    const int32_t hop = meta.hop_length;
    const int32_t win = meta.win_length;
    const int32_t num_bins = meta.num_mel_bins;
    const float preemph = meta.preemphasis;

    std::vector<float> ans(static_cast<int64_t>(num_out) * num_bins);

    int32_t num_valid = std::min(num_out, num_frames - start);
    if (num_valid <= 0) {
      return ans;
    }

    // Samples [begin, begin + len) of the preemphasized waveform, with zeros
    // outside [0, n)
    int64_t begin = static_cast<int64_t>(start) * hop - win / 2;
    int64_t len = static_cast<int64_t>(num_valid - 1) * hop + win;
    std::vector<float> samples(len);
    for (int64_t i = 0; i != len; ++i) {
      int64_t t = begin + i;
      if (t < 0 || t >= n) {
        continue;
      }
      samples[i] = t == 0 ? audio[0] : audio[t] - preemph * audio[t - 1];
    }

    knf::OnlineFbank fbank(fbank_opts_);
    fbank.AcceptWaveform(meta.sample_rate, samples.data(),
                         static_cast<int32_t>(samples.size()));
    fbank.InputFinished();

    if (fbank.NumFramesReady() < num_valid) {
      SHERPA_ONNX_LOGE("Expect %d feature frames. Got: %d", num_valid,
                       fbank.NumFramesReady());
      SHERPA_ONNX_EXIT(-1);
    }

    // torch.log(mel_spec + 2**-24)
    constexpr float kLogGuard = 5.960464477539063e-08f;
    for (int32_t i = 0; i != num_valid; ++i) {
      const float *f = fbank.GetFrame(i);
      float *dst = ans.data() + static_cast<int64_t>(i) * num_bins;
      for (int32_t j = 0; j != num_bins; ++j) {
        dst[j] = std::log(f[j] + kLogGuard);
      }
    }

    return ans;
  }

  OfflineSpeakerDiarizationResult ComputeResult(
      const std::vector<float> &probs) const {
    const auto &meta = model_.GetModelMetaData();
    const int32_t num_speakers = meta.cache.num_speakers;
    const int32_t num_frames =
        static_cast<int32_t>(probs.size()) / num_speakers;
    const float threshold = config_.segmentation.sortformer.threshold;
    const float frame_shift =
        static_cast<float>(meta.hop_length) / meta.sample_rate;

    OfflineSpeakerDiarizationResult ans;

    // Speaker IDs are compacted, keeping the order of first arrival
    int32_t num_found = 0;

    for (int32_t s = 0; s != num_speakers; ++s) {
      std::vector<OfflineSpeakerDiarizationSegment> this_speaker;

      int32_t start_index = -1;
      for (int32_t f = 0; f <= num_frames; ++f) {
        bool is_active =
            f < num_frames && probs[f * num_speakers + s] > threshold;
        if (is_active && start_index < 0) {
          start_index = f;
        } else if (!is_active && start_index >= 0) {
          this_speaker.emplace_back(start_index * frame_shift, f * frame_shift,
                                    num_found);
          start_index = -1;
        }
      }

      // merge segments if the gap between them is less than min_duration_off
      MergeSegments(&this_speaker);

      bool found = false;
      for (const auto &seg : this_speaker) {
        if (seg.Duration() > config_.min_duration_on) {
          ans.Add(seg);
          found = true;
        }
      }

      num_found += found;
    }

    return ans;
  }

  void MergeSegments(
      std::vector<OfflineSpeakerDiarizationSegment> *segments) const {
    float min_duration_off = config_.min_duration_off;
    bool changed = true;
    while (changed) {
      changed = false;
      for (int32_t i = 0; i < static_cast<int32_t>(segments->size()) - 1; ++i) {
        auto s = (*segments)[i].Merge((*segments)[i + 1], min_duration_off);
        if (s) {
          (*segments)[i] = s.value();
          segments->erase(segments->begin() + i + 1);

          changed = true;
          break;
        }
      }
    }
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationSortformerModel model_;
  knf::FbankOptions fbank_opts_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_SORTFORMER_IMPL_H_
