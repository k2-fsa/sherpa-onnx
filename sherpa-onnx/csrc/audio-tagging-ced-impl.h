// sherpa-onnx/csrc/audio-tagging-ced-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_AUDIO_TAGGING_CED_IMPL_H_
#define SHERPA_ONNX_CSRC_AUDIO_TAGGING_CED_IMPL_H_

#include <assert.h>

#include <memory>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/audio-tagging-impl.h"
#include "sherpa-onnx/csrc/audio-tagging-label-file.h"
#include "sherpa-onnx/csrc/audio-tagging.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-ced-model.h"

namespace sherpa_onnx {

class AudioTaggingCEDImpl : public AudioTaggingImpl {
 public:
  explicit AudioTaggingCEDImpl(const AudioTaggingConfig &config)
      : config_(config), model_(config.model), labels_(config.labels) {
    if (model_.NumEventClasses() != labels_.NumEventClasses()) {
      SHERPA_ONNX_LOGE("number of classes: %d (model) != %d (label file)",
                       model_.NumEventClasses(), labels_.NumEventClasses());
      exit(-1);
    }
  }

#if __ANDROID_API__ >= 9
  explicit AudioTaggingCEDImpl(AAssetManager *mgr,
                               const AudioTaggingConfig &config)
      : config_(config),
        model_(mgr, config.model),
        labels_(mgr, config.labels) {
    if (model_.NumEventClasses() != labels_.NumEventClasses()) {
      SHERPA_ONNX_LOGE("number of classes: %d (model) != %d (label file)",
                       model_.NumEventClasses(), labels_.NumEventClasses());
      exit(-1);
    }
  }
#endif

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(CEDTag{});
  }

  std::vector<AudioEvent> Compute(OfflineStream *s,
                                  int32_t top_k = -1) const override {
    if (top_k < 0) {
      top_k = config_.top_k;
    }

    int32_t num_event_classes = model_.NumEventClasses();

    if (top_k > num_event_classes) {
      top_k = num_event_classes;
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // WARNING(fangjun): It is fixed to 64 for CED models
    int32_t feat_dim = 64;
    std::vector<float> f = s->GetFrames();

    int32_t num_frames = f.size() / feat_dim;
    assert(feat_dim * num_frames == static_cast<int32_t>(f.size()));

    std::array<int64_t, 3> shape = {1, num_frames, feat_dim};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    Ort::Value probs = model_.Forward(std::move(x));

    const float *p = probs.GetTensorData<float>();

    std::vector<int32_t> top_k_indexes = TopkIndex(p, num_event_classes, top_k);

    std::vector<AudioEvent> ans(top_k);

    int32_t i = 0;

    for (int32_t index : top_k_indexes) {
      ans[i].name = labels_.GetEventName(index);
      ans[i].index = index;
      ans[i].prob = p[index];
      i += 1;
    }

    return ans;
  }

 private:
  AudioTaggingConfig config_;
  OfflineCEDModel model_;
  AudioTaggingLabels labels_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_AUDIO_TAGGING_CED_IMPL_H_
