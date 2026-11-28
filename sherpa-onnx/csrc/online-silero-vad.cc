// sherpa-onnx/csrc/online-silero-vad.cc
//
// Copyright (c)  2026  Manickavela Arumugam

#include "sherpa-onnx/csrc/online-silero-vad.h"

#include <algorithm>
#include <array>
#include <utility>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

// Silero is tiny; the session is pinned to CPU like the offline model.
static Ort::SessionOptions GetSileroSessionOptions(int32_t num_threads) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(num_threads);
  so.SetInterOpNumThreads(1);
  so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  return so;
}

OnlineSileroVad::OnlineSileroVad(const OnlineVadModelConfig &config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_ERROR),
      sess_opts_(GetSileroSessionOptions(config.num_threads)),
      allocator_{} {
  if (config_.sample_rate != 16000) {
    SHERPA_ONNX_LOGE("--online-vad-sample-rate: expected 16000. Given: %d",
                     config_.sample_rate);
    SHERPA_ONNX_EXIT(-1);
  }

  Init();

  window_shift_ = config_.window_size;
  window_size_ = config_.window_size + window_overlap_;
  min_silence_samples_ = static_cast<int32_t>(config_.sample_rate *
                                              config_.min_silence_duration);
  min_speech_samples_ = static_cast<int32_t>(config_.sample_rate *
                                             config_.min_speech_duration);
  neg_threshold_ = config_.neg_threshold < 0
                       ? std::max(config_.threshold - 0.15f, 0.01f)
                       : std::max(config_.neg_threshold, 0.01f);

  SHERPA_ONNX_LOGE(
      "[ONLINE-VAD] silero (shared session): v5=%d threshold=%.2f "
      "window_size=%d window_shift=%d overlap=%d sample_rate=%d",
      is_v5_ ? 1 : 0, config_.threshold, window_size_, window_shift_,
      window_overlap_, config_.sample_rate);
}

OnlineSileroVad::~OnlineSileroVad() = default;

void OnlineSileroVad::Init() {
  auto buf = ReadFile(config_.model);
  sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                         sess_opts_);
  GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
  GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

  if (input_names_.size() == 4 && output_names_.size() == 3) {
    is_v5_ = false;
    window_overlap_ = 0;
  } else if (input_names_.size() == 3 && output_names_.size() == 2) {
    is_v5_ = true;
    // 64 samples of context at 16 kHz.
    window_overlap_ = 64;
    if (config_.window_size != 512) {
      SHERPA_ONNX_LOGE(
          "For silero_vad v5, we require --online-vad-window-size=512");
      SHERPA_ONNX_EXIT(-1);
    }
  } else {
    SHERPA_ONNX_LOGE("Unsupported silero vad model (inputs=%d outputs=%d)",
                     static_cast<int32_t>(input_names_.size()),
                     static_cast<int32_t>(output_names_.size()));
    SHERPA_ONNX_EXIT(-1);
  }
}

std::vector<Ort::Value> OnlineSileroVad::GetInitStates() {
  std::vector<Ort::Value> states;
  if (is_v5_) {
    std::array<int64_t, 3> shape{2, 1, 128};
    auto s = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                             shape.size());
    Fill<float>(&s, 0);
    states.push_back(std::move(s));
  } else {
    std::array<int64_t, 3> shape{2, 1, 64};
    auto h = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                             shape.size());
    auto c = Ort::Value::CreateTensor<float>(allocator_, shape.data(),
                                             shape.size());
    Fill<float>(&h, 0);
    Fill<float>(&c, 0);
    states.push_back(std::move(h));
    states.push_back(std::move(c));
  }
  return states;
}

std::vector<Ort::Value> OnlineSileroVad::StackStates(
    const std::vector<int32_t> &idx, OnlineStream **ss) {
  const int32_t batch = static_cast<int32_t>(idx.size());
  const auto &first = ss[idx[0]]->GetVadStates();
  const int32_t num_tensors = static_cast<int32_t>(first.size());

  std::vector<Ort::Value> stacked;
  stacked.reserve(num_tensors);

  for (int32_t t = 0; t < num_tensors; ++t) {
    auto shape = first[t].GetTensorTypeAndShapeInfo().GetShape();  // [2,1,H]
    const int32_t layers = static_cast<int32_t>(shape[0]);
    const int32_t hid = static_cast<int32_t>(shape[2]);
    std::array<int64_t, 3> out_shape{layers, batch, hid};
    auto dst = Ort::Value::CreateTensor<float>(allocator_, out_shape.data(),
                                               out_shape.size());
    float *dst_data = dst.GetTensorMutableData<float>();
    for (int32_t j = 0; j < batch; ++j) {
      const float *src = ss[idx[j]]->GetVadStates()[t].GetTensorData<float>();
      for (int32_t l = 0; l < layers; ++l) {
        std::copy(src + l * hid, src + (l + 1) * hid,
                  dst_data + l * batch * hid + j * hid);
      }
    }
    stacked.push_back(std::move(dst));
  }
  return stacked;
}

void OnlineSileroVad::UnStackStates(std::vector<Ort::Value> &stacked,
                                    const std::vector<int32_t> &idx,
                                    OnlineStream **ss) {
  const int32_t batch = static_cast<int32_t>(idx.size());
  const int32_t num_tensors = static_cast<int32_t>(stacked.size());

  std::vector<std::vector<Ort::Value>> per_stream(batch);
  for (int32_t j = 0; j < batch; ++j) per_stream[j].reserve(num_tensors);

  for (int32_t t = 0; t < num_tensors; ++t) {
    auto shape = stacked[t].GetTensorTypeAndShapeInfo().GetShape();  // [2,B,H]
    const int32_t layers = static_cast<int32_t>(shape[0]);
    const int32_t hid = static_cast<int32_t>(shape[2]);
    const float *src = stacked[t].GetTensorData<float>();
    for (int32_t j = 0; j < batch; ++j) {
      std::array<int64_t, 3> one_shape{layers, 1, hid};
      auto dst = Ort::Value::CreateTensor<float>(allocator_, one_shape.data(),
                                                 one_shape.size());
      float *dst_data = dst.GetTensorMutableData<float>();
      for (int32_t l = 0; l < layers; ++l) {
        std::copy(src + l * batch * hid + j * hid,
                  src + l * batch * hid + (j + 1) * hid, dst_data + l * hid);
      }
      per_stream[j].push_back(std::move(dst));
    }
  }
  for (int32_t j = 0; j < batch; ++j) {
    ss[idx[j]]->SetVadStates(std::move(per_stream[j]));
  }
}

std::vector<Ort::Value> OnlineSileroVad::RunBatch(
    const float *x, int32_t batch_size, std::vector<Ort::Value> &stacked_state,
    std::vector<float> *probs, const Ort::MemoryInfo &memory_info) {
  std::array<int64_t, 2> x_shape{batch_size, window_size_};
  Ort::Value x_t = Ort::Value::CreateTensor(
      memory_info, const_cast<float *>(x),
      static_cast<size_t>(batch_size) * window_size_, x_shape.data(),
      x_shape.size());

  int64_t sr = config_.sample_rate;
  std::array<int64_t, 1> sr_shape{1};
  Ort::Value sr_t =
      Ort::Value::CreateTensor(memory_info, &sr, 1, sr_shape.data(), 1);

  std::vector<Ort::Value> inputs;
  if (is_v5_) {
    // order: input, state, sr
    inputs.reserve(3);
    inputs.push_back(std::move(x_t));
    inputs.push_back(std::move(stacked_state[0]));
    inputs.push_back(std::move(sr_t));
  } else {
    // order: input, sr, h, c
    inputs.reserve(4);
    inputs.push_back(std::move(x_t));
    inputs.push_back(std::move(sr_t));
    inputs.push_back(std::move(stacked_state[0]));
    inputs.push_back(std::move(stacked_state[1]));
  }

  auto out = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                        inputs.size(), output_names_ptr_.data(),
                        output_names_ptr_.size());

  // out[0] = probabilities [B, 1]
  const float *p = out[0].GetTensorData<float>();
  probs->assign(p, p + batch_size);

  // remaining outputs are the new state tensors (v5: 1, v4: 2)
  std::vector<Ort::Value> new_state;
  new_state.reserve(out.size() - 1);
  for (size_t k = 1; k < out.size(); ++k) {
    new_state.push_back(std::move(out[k]));
  }
  return new_state;
}

bool OnlineSileroVad::Decide(float prob, bool *triggered,
                             int32_t *current_sample, int32_t *temp_start,
                             int32_t *temp_end) const {
  const float threshold = config_.threshold;
  *current_sample += window_shift_;

  if (prob > threshold && *temp_end != 0) {
    *temp_end = 0;
  }

  if (prob > threshold && *temp_start == 0) {
    // start speaking, but it must satisfy min_speech_duration
    *temp_start = *current_sample;
    return false;
  }

  if (prob > threshold && *temp_start != 0 && !*triggered) {
    if (*current_sample - *temp_start < min_speech_samples_) {
      return false;
    }
    *triggered = true;
    return true;
  }

  if (prob < threshold && !*triggered) {
    *temp_start = 0;
    *temp_end = 0;
    return false;
  }

  if (prob > neg_threshold_ && *triggered) {
    return true;
  }

  if (prob > threshold && !*triggered) {
    *triggered = true;
    return true;
  }

  if (prob < threshold && *triggered) {
    if (*temp_end == 0) {
      *temp_end = *current_sample;
    }
    if (*current_sample - *temp_end < min_silence_samples_) {
      // still inside the min-silence hangover
      return true;
    }
    *temp_start = 0;
    *temp_end = 0;
    *triggered = false;
    return false;
  }

  return false;
}

void OnlineSileroVad::WarmUp(int32_t n) {
  if (n <= 0 || window_size_ <= 0) return;
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  auto state = GetInitStates();  // [2,1,H] == [2,B=1,H]
  std::vector<float> x(static_cast<size_t>(window_size_), 0.0f);
  std::vector<float> probs;
  for (int32_t i = 0; i < n; ++i) {
    state = RunBatch(x.data(), /*batch_size=*/1, state, &probs, memory_info);
  }
}

void OnlineSileroVad::RunVad(OnlineStream **ss, int32_t n) {
  if (window_size_ <= 0 || window_shift_ <= 0) return;

  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Re-batch each round: streams with uneven amounts of pending audio drop
  // out as soon as they no longer hold a full window.
  for (;;) {
    std::vector<int32_t> idx;
    for (int32_t i = 0; i < n; ++i) {
      if (ss[i]->VadPendingSize() < window_size_) continue;
      if (ss[i]->GetVadStates().empty()) {
        ss[i]->SetVadStates(GetInitStates());
      }
      idx.push_back(i);
    }
    if (idx.empty()) break;

    const int32_t batch = static_cast<int32_t>(idx.size());

    // Assemble the batched input [B, window_size_]. The window is peeked
    // (copied under the stream lock) and window_shift_ samples are dropped
    // only after the inference succeeded.
    std::vector<float> x(static_cast<size_t>(batch) * window_size_);
    std::vector<float> win;
    for (int32_t j = 0; j < batch; ++j) {
      ss[idx[j]]->VadPeekWindow(window_size_, &win);
      std::copy(win.begin(), win.end(),
                x.begin() + static_cast<size_t>(j) * window_size_);
    }

    auto stacked = StackStates(idx, ss);
    std::vector<float> probs;
    auto new_state = RunBatch(x.data(), batch, stacked, &probs, memory_info);
    UnStackStates(new_state, idx, ss);

    for (int32_t j = 0; j < batch; ++j) {
      OnlineStream *s = ss[idx[j]];
      bool is_speech = Decide(probs[j], &s->IsVadTriggered(),
                              &s->GetVadCurrentSample(), &s->GetVadTempStart(),
                              &s->GetVadTempEnd());
      s->VadDropSamples(window_shift_);
      // Records the prediction and, on a confirmed speech -> silence flip,
      // arms the stream's endpoint + end-of-utterance flush.
      s->VadPushPrediction(is_speech);
    }
  }
}

}  // namespace sherpa_onnx
