// sherpa-onnx/csrc/vad-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VAD_MODEL_H_
#define SHERPA_ONNX_CSRC_VAD_MODEL_H_

#include <cstring>
#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

// The state a VadModel updates for each window. Save it with GetState()
// and restore it with SetState() so one model can serve several streams
// alternately: each stream keeps its own state between its windows.
struct VadModelState {
  // Recurrent model states, e.g. h and c for silero v4. states[i] holds
  // the values of state i flattened in row major and state_shapes[i] is
  // its shape. Both have the same length.
  std::vector<std::vector<float>> states;
  std::vector<std::vector<int64_t>> state_shapes;

  // Counters used by IsSpeech() to smooth single windows into speech and
  // silence runs.
  bool triggered = false;
  int32_t current_sample = 0;
  int32_t temp_start = 0;
  int32_t temp_end = 0;

  // The last sample and features of ten-vad, which span windows.
  float last_sample = 0;
  std::vector<float> last_features;
};

// Helpers for subclasses whose recurrent states are Ort::Value tensors.

inline void CopyToModelState(const std::vector<Ort::Value> &src,
                             VadModelState *dst) {
  dst->states.clear();
  dst->state_shapes.clear();
  for (const auto &s : src) {
    auto info = s.GetTensorTypeAndShapeInfo();
    dst->state_shapes.push_back(info.GetShape());
    const float *p = s.GetTensorData<float>();
    dst->states.emplace_back(p, p + info.GetElementCount());
  }
}

inline void CopyFromModelState(const VadModelState &src,
                               std::vector<Ort::Value> *dst,
                               OrtAllocator *allocator) {
  dst->clear();
  dst->reserve(src.states.size());
  for (size_t i = 0; i < src.states.size(); ++i) {
    const auto &shape = src.state_shapes[i];
    Ort::Value v = Ort::Value::CreateTensor(
        allocator, const_cast<int64_t *>(shape.data()), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    std::memcpy(v.GetTensorMutableData<float>(), src.states[i].data(),
                src.states[i].size() * sizeof(float));
    dst->push_back(std::move(v));
  }
}

class VadModel {
 public:
  virtual ~VadModel() = default;

  static std::unique_ptr<VadModel> Create(const VadModelConfig &config);

  template <typename Manager>
  static std::unique_ptr<VadModel> Create(Manager *mgr,
                                          const VadModelConfig &config);

  // reset the internal model states
  virtual void Reset() = 0;

  /**
   * @param samples Pointer to a 1-d array containing audio samples.
   *                Each sample should be normalized to the range [-1, 1].
   * @param n Number of samples. Should be equal to WindowSize()
   *
   * @return Return true if speech is detected. Return false otherwise.
   */
  virtual bool IsSpeech(const float *samples, int32_t n) = 0;

  virtual float Compute(const float *samples, int32_t n) = 0;

  virtual int32_t WindowSize() const = 0;

  virtual int32_t WindowShift() const = 0;

  virtual int32_t MinSilenceDurationSamples() const = 0;
  virtual int32_t MinSpeechDurationSamples() const = 0;
  virtual void SetMinSilenceDuration(float s) = 0;
  virtual void SetThreshold(float threshold) = 0;

  // Copy the current state out. The model keeps its state, so it can be
  // used as before afterwards.
  virtual VadModelState GetState() const = 0;

  // Replace the current state. state is copied, so the caller is free to
  // discard it afterwards.
  virtual void SetState(VadModelState state) = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VAD_MODEL_H_
