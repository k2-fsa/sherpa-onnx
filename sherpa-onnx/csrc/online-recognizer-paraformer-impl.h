// sherpa-onnx/csrc/online-recognizer-paraformer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-lm.h"
#include "sherpa-onnx/csrc/online-paraformer-model.h"
#include "sherpa-onnx/csrc/online-recognizer-impl.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

class OnlineRecognizerParaformerImpl : public OnlineRecognizerImpl {
 public:
  explicit OnlineRecognizerParaformerImpl(const OnlineRecognizerConfig &config)
      : config_(config),
        model_(config.model_config),
        sym_(config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (config.decoding_method == "greedy_search") {
      // add greedy search decoder
      // SHERPA_ONNX_LOGE("to be implemented");
      // exit(-1);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    // Paraformer models assume input samples are in the range
    // [-32768, 32767], so we set normalize_samples to false
    config_.feat_config.normalize_samples = false;
  }

#if __ANDROID_API__ >= 9
  explicit OnlineRecognizerParaformerImpl(AAssetManager *mgr,
                                          const OnlineRecognizerConfig &config)
      : config_(config),
        model_(mgr, config.model_config),
        sym_(mgr, config.model_config.tokens),
        endpoint_(config_.endpoint_config) {
    if (config.decoding_method == "greedy_search") {
      // add greedy search decoder
      // SHERPA_ONNX_LOGE("to be implemented");
      // exit(-1);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    // Paraformer models assume input samples are in the range
    // [-32768, 32767], so we set normalize_samples to false
    config_.feat_config.normalize_samples = false;
  }
#endif
  OnlineRecognizerParaformerImpl(const OnlineRecognizerParaformerImpl &) =
      delete;

  OnlineRecognizerParaformerImpl operator=(
      const OnlineRecognizerParaformerImpl &) = delete;

  std::unique_ptr<OnlineStream> CreateStream() const override {
    auto stream = std::make_unique<OnlineStream>(config_.feat_config);
    return stream;
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + chunk_size_ < s->NumFramesReady();
  }

  void DecodeStreams(OnlineStream **ss, int32_t n) const override {
    // TODO(fangjun): Support batch size > 1
    for (int32_t i = 0; i != n; ++i) {
      DecodeStream(ss[i]);
    }
  }

  OnlineRecognizerResult GetResult(OnlineStream *s) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
    return {};
  }

  bool IsEndpoint(OnlineStream *s) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
    return false;
  }

  void Reset(OnlineStream *s) const override {
    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

 private:
  void DecodeStream(OnlineStream *s) const {
    SHERPA_ONNX_LOGE("NumFramesReady: %d\n", s->NumFramesReady());

    const auto num_processed_frames = s->GetNumProcessedFrames();
    std::vector<float> frames = s->GetFrames(num_processed_frames, chunk_size_);
    s->GetNumProcessedFrames() += chunk_size_;

    frames = ApplyLFR(frames);
    ApplyCMVN(&frames);

    // We have scaled inv_stddev by sqrt(encoder_output_size)
    // so the following line can be commented out
    // frames *= encoder_output_size ** 0.5

    // TODO(fangjun): Implement positional embedding

    SHERPA_ONNX_LOGE("to be implemented");
    exit(-1);
  }

  std::vector<float> ApplyLFR(const std::vector<float> &in) const {
    int32_t lfr_window_size = model_.LfrWindowSize();
    int32_t lfr_window_shift = model_.LfrWindowShift();
    int32_t in_feat_dim = config_.feat_config.feature_dim;

    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void ApplyCMVN(std::vector<float> *v) const {
    const std::vector<float> &neg_mean = model_.NegativeMean();
    const std::vector<float> &inv_stddev = model_.InverseStdDev();

    int32_t dim = neg_mean.size();
    int32_t num_frames = v->size() / dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        p[k] = (p[k] + neg_mean[k]) * inv_stddev[k];
      }

      p += dim;
    }
  }

 private:
  OnlineRecognizerConfig config_;
  OnlineParaformerModel model_;
  SymbolTable sym_;
  Endpoint endpoint_;

  // 0.6 seconds
  int32_t chunk_size_ = 61;
  // (61 - 7) / 6 + 1 = 10
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_PARAFORMER_IMPL_H_
