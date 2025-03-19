// sherpa-onnx/csrc/offline-lm.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-lm.h"

#include <algorithm>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/lodr-fst.h"
#include "sherpa-onnx/csrc/offline-rnn-lm.h"

namespace sherpa_onnx {

std::unique_ptr<OfflineLM> OfflineLM::Create(const OfflineLMConfig &config) {
  return std::make_unique<OfflineRnnLM>(config);
}

template <typename Manager>
std::unique_ptr<OfflineLM> OfflineLM::Create(Manager *mgr,
                                             const OfflineLMConfig &config) {
  return std::make_unique<OfflineRnnLM>(mgr, config);
}

void OfflineLM::ComputeLMScore(float scale, int32_t context_size,
                               std::vector<Hypotheses> *hyps) {
  // compute the max token seq so that we know how much space to allocate
  int32_t max_token_seq = 0;
  int32_t num_hyps = 0;

  // we subtract context_size below since each token sequence is prepended
  // with context_size blanks
  for (const auto &h : *hyps) {
    num_hyps += h.Size();
    for (const auto &t : h) {
      max_token_seq =
          std::max<int32_t>(max_token_seq, t.second.ys.size() - context_size);
    }
  }

  Ort::AllocatorWithDefaultOptions allocator;
  std::array<int64_t, 2> x_shape{num_hyps, max_token_seq};
  Ort::Value x = Ort::Value::CreateTensor<int64_t>(allocator, x_shape.data(),
                                                   x_shape.size());

  std::array<int64_t, 1> x_lens_shape{num_hyps};
  Ort::Value x_lens = Ort::Value::CreateTensor<int64_t>(
      allocator, x_lens_shape.data(), x_lens_shape.size());

  int64_t *p = x.GetTensorMutableData<int64_t>();
  std::fill(p, p + num_hyps * max_token_seq, 0);

  int64_t *p_lens = x_lens.GetTensorMutableData<int64_t>();

  for (const auto &h : *hyps) {
    for (const auto &t : h) {
      const auto &ys = t.second.ys;
      int32_t len = ys.size() - context_size;
      std::copy(ys.begin() + context_size, ys.end(), p);
      *p_lens = len;

      p += max_token_seq;
      ++p_lens;
    }
  }
  auto negative_loglike = Rescore(std::move(x), std::move(x_lens));
  const float *p_nll = negative_loglike.GetTensorData<float>();
  // We scale LODR scale with LM scale to replicate Icefall code
  auto lodr_scale = config_.lodr_scale * scale;
  for (auto &h : *hyps) {
    for (auto &t : h) {
      // Use -scale here since we want to change negative loglike to loglike.
      t.second.lm_log_prob = -scale * (*p_nll);
      ++p_nll;
      // apply LODR to hyp score
      if (lodr_fst_ != nullptr) {
        lodr_fst_->ComputeScore(lodr_scale, &t.second, context_size);
      }
    }
  }
}

#if __ANDROID_API__ >= 9
template std::unique_ptr<OfflineLM> OfflineLM::Create(
    AAssetManager *mgr, const OfflineLMConfig &config);
#endif

#if __OHOS__
template std::unique_ptr<OfflineLM> OfflineLM::Create(
    NativeResourceManager *mgr, const OfflineLMConfig &config);
#endif

}  // namespace sherpa_onnx
