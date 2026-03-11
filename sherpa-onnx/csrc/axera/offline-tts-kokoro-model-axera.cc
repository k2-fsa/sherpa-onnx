// sherpa-onnx/csrc/axera/offline-tts-kokoro-model-axera.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axera/offline-tts-kokoro-model-axera.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "ax_engine_api.h"  // NOLINT
#include "ax_sys_api.h"     // NOLINT
#include "kaldi-native-fbank/csrc/istft.h"
#include "sherpa-onnx/csrc/axera/ax-engine-guard.h"
#include "sherpa-onnx/csrc/axera/utils.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

static std::vector<float> Sigmoid(const std::vector<float> &x) {
  std::vector<float> result(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    result[i] = 1.0f / (1.0f + std::exp(-x[i]));
  }
  return result;
}

template <typename T>
static std::vector<size_t> Argsort(const std::vector<T> &v, int32_t len,
                                   bool reverse) {
  std::vector<size_t> idx(len);
  std::iota(idx.begin(), idx.end(), 0);

  if (!reverse) {
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  } else {
    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });
  }

  return idx;
}

template <typename T>
static std::vector<T> NpRepeat(const std::vector<T> &v,
                               const std::vector<int32_t> &times) {
  std::vector<T> result;
  for (size_t i = 0; i < times.size(); i++) {
    for (int32_t n = 0; n < times[i]; n++) {
      result.push_back(v[i]);
    }
  }
  return result;
}

class AxeraModel {
 public:
  explicit AxeraModel(const std::string &filename) {
    auto buf = ReadFile(filename);
    Init(buf.data(), buf.size());
  }

  AxeraModel(const void *cpu_buf, size_t buf_len_in_bytes) {
    Init(cpu_buf, buf_len_in_bytes);
  }

  ~AxeraModel() {
    if (io_data_.pInputs || io_data_.pOutputs) {
      FreeIO(&io_data_);
    }

    if (handle_) {
      AX_ENGINE_DestroyHandle(handle_);
    }
  }

  std::vector<int32_t> TensorShape(const std::string &name) const {
    auto it = input_name_to_index_.find(name);
    if (it != input_name_to_index_.end()) {
      return input_tensor_shapes_[it->second];
    }

    auto it2 = output_name_to_index_.find(name);
    if (it2 != output_name_to_index_.end()) {
      return output_tensor_shapes_[it2->second];
    }

    SHERPA_ONNX_LOGE("Found no tensor with name: '%s'", name.c_str());
    return {};
  }

  template <typename T>
  bool SetInputTensorData(const std::string &name, const T *p, int32_t n) {
    auto it = input_name_to_index_.find(name);
    if (it == input_name_to_index_.end()) {
      SHERPA_ONNX_LOGE("Found no input tensor with name: '%s'", name.c_str());
      return false;
    }

    int32_t i = it->second;
    size_t expected_size = io_info_->pInputs[i].nSize;
    size_t given_size = n * sizeof(T);

    if (expected_size != given_size) {
      SHERPA_ONNX_LOGE(
          "Input tensor '%s' size mismatch. Expected %zu bytes, got %zu bytes",
          name.c_str(), expected_size, given_size);
      return false;
    }

    std::memcpy(io_data_.pInputs[i].pVirAddr, p, expected_size);
    return true;
  }

  std::vector<float> GetOutputTensorData(const std::string &name) {
    auto it = output_name_to_index_.find(name);
    if (it == output_name_to_index_.end()) {
      SHERPA_ONNX_LOGE("Found no output tensor with name: '%s'", name.c_str());
      return {};
    }

    int32_t i = it->second;
    const auto &out_meta = io_info_->pOutputs[i];
    auto &out_buf = io_data_.pOutputs[i];

    AX_SYS_MinvalidateCache(out_buf.phyAddr, out_buf.pVirAddr, out_meta.nSize);

    size_t out_elems = out_meta.nSize / sizeof(float);
    std::vector<float> out(out_elems);
    std::memcpy(out.data(), out_buf.pVirAddr, out_meta.nSize);

    return out;
  }

  bool Run() {
    auto ret = AX_ENGINE_RunSync(handle_, &io_data_);
    if (ret != 0) {
      SHERPA_ONNX_LOGE("AX_ENGINE_RunSync failed, ret = %d", ret);
      return false;
    }

    return true;
  }

 private:
  void Init(const void *model_data, size_t model_data_length) {
    InitContext(model_data, model_data_length, false, &handle_);
    InitInputOutputAttrs(handle_, false, &io_info_);
    PrepareIO(io_info_, &io_data_, false);

    input_tensor_shapes_.reserve(io_info_->nInputSize);
    for (uint32_t i = 0; i < io_info_->nInputSize; ++i) {
      const auto &in = io_info_->pInputs[i];
      std::string name = in.pName;
      input_name_to_index_[name] = i;
      input_tensor_shapes_.emplace_back(in.pShape, in.pShape + in.nShapeSize);
    }

    output_tensor_shapes_.reserve(io_info_->nOutputSize);
    for (uint32_t i = 0; i < io_info_->nOutputSize; ++i) {
      const auto &out = io_info_->pOutputs[i];
      std::string name = out.pName;
      output_name_to_index_[name] = i;
      output_tensor_shapes_.emplace_back(out.pShape,
                                         out.pShape + out.nShapeSize);
    }
  }

 private:
  AX_ENGINE_HANDLE handle_ = nullptr;
  AX_ENGINE_IO_INFO_T *io_info_ = nullptr;
  AX_ENGINE_IO_T io_data_{};

  std::unordered_map<std::string, int32_t> input_name_to_index_;
  std::unordered_map<std::string, int32_t> output_name_to_index_;

  std::vector<std::vector<int32_t>> input_tensor_shapes_;
  std::vector<std::vector<int32_t>> output_tensor_shapes_;
};

static std::string GetModelPath(const std::string &model, const std::string &p,
                                bool is_onnx = false) {
  std::string suffix = is_onnx ? ".onnx" : ".axmodel";

  std::string path1 = model + "/kokoro_" + p + suffix;
  if (FileExists(path1)) {
    return path1;
  }

  return model + "_" + p + suffix;
}

class OfflineTtsKokoroModelAxera::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(config);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(mgr, config);
  }

  const OfflineTtsKokoroModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  std::vector<float> Run(std::vector<int64_t> input_ids, int64_t sid,
                         float speed) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (input_ids.size() < 2) {
      return {};
    }

    if (config_.kokoro.length_scale != 1 && speed == 1) {
      speed = 1. / config_.kokoro.length_scale;
    }

    int32_t phoneme_len = static_cast<int32_t>(input_ids.size()) - 2;
    if (phoneme_len < 0) phoneme_len = 0;

    std::vector<float> ref_s =
        LoadVoiceEmbedding(static_cast<int32_t>(sid), phoneme_len);

    std::vector<float> audio;
    bool success = RunModels(input_ids, ref_s, speed, audio);
    if (!success) {
      SHERPA_ONNX_LOGE("Run models failed!");
      return {};
    }

    return audio;
  }

 private:
  void Init(const OfflineTtsModelConfig &config) {
    std::string model_str = config.kokoro.model;

    std::vector<std::string> model_files;
    SplitStringToVector(model_str, ",", false, &model_files);

    std::vector<std::string> model_paths;

    if (config_.provider == "axera") {
      // axera provider requires exactly 4 model files
      if (model_files.size() != 4) {
        SHERPA_ONNX_LOGE(
            "For axera provider, model should contain exactly 4 files "
            "separated by comma. Given %d files: %s",
            static_cast<int32_t>(model_files.size()), model_str.c_str());
        SHERPA_ONNX_EXIT(-1);
      }

      // Validate all 4 files exist
      for (int32_t i = 0; i < 4; ++i) {
        if (!FileExists(model_files[i])) {
          SHERPA_ONNX_LOGE("Model file '%s' does not exist",
                           model_files[i].c_str());
          SHERPA_ONNX_EXIT(-1);
        }
      }
      model_paths = model_files;
    } else {
      SHERPA_ONNX_LOGE(
          "This model only supports axera provider. Please use provider=axera");
      SHERPA_ONNX_EXIT(-1);
    }

    // Load axera models (indices 0, 1, 2)
    for (int32_t i = 0; i < 3; ++i) {
      axera_models_.push_back(std::make_unique<AxeraModel>(model_paths[i]));
    }

    // Load onnx model (index 3)
    auto buf4 = ReadFile(model_paths[3]);
    model4_ = std::make_unique<Ort::Session>(env_, buf4.data(), buf4.size(),
                                             sess_opts_);
    GetInputNames(model4_.get(), &model4_input_names_,
                  &model4_input_names_ptr_);
    GetOutputNames(model4_.get(), &model4_output_names_,
                   &model4_output_names_ptr_);

    auto voices_buf = ReadFile(config.kokoro.voices);
    LoadVoices(voices_buf.data(), voices_buf.size());
  }

  template <typename Manager>
  void Init(Manager *mgr, const OfflineTtsModelConfig &config) {
    std::string model_str = config.kokoro.model;

    std::vector<std::string> model_files;
    SplitStringToVector(model_str, ",", false, &model_files);

    std::vector<std::string> model_paths;

    if (config_.provider == "axera") {
      // axera provider requires exactly 4 model files
      if (model_files.size() != 4) {
        SHERPA_ONNX_LOGE(
            "For axera provider, model should contain exactly 4 files "
            "separated by comma. Given %d files: %s",
            static_cast<int32_t>(model_files.size()), model_str.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      model_paths = model_files;
    } else {
      SHERPA_ONNX_LOGE(
          "This model only supports axera provider. Please use provider=axera");
      SHERPA_ONNX_EXIT(-1);
    }

    // Load axera models (indices 0, 1, 2)
    for (int32_t i = 0; i < 3; ++i) {
      auto buf = ReadFile(mgr, model_paths[i]);
      axera_models_.push_back(
          std::make_unique<AxeraModel>(buf.data(), buf.size()));
    }

    // Load onnx model (index 3)
    auto buf4 = ReadFile(mgr, model_paths[3]);
    model4_ = std::make_unique<Ort::Session>(env_, buf4.data(), buf4.size(),
                                             sess_opts_);
    GetInputNames(model4_.get(), &model4_input_names_,
                  &model4_input_names_ptr_);
    GetOutputNames(model4_.get(), &model4_output_names_,
                   &model4_output_names_ptr_);

    auto voices_buf = ReadFile(mgr, config.kokoro.voices);
    LoadVoices(voices_buf.data(), voices_buf.size());
  }

  void LoadVoices(const char *voices_data, size_t voices_data_length) {
    // Get metadata from model4 (ONNX model)
    Ort::ModelMetadata meta_data = model4_->GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;

    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(meta_data_.version, "version", 1);
    SHERPA_ONNX_READ_META_DATA(meta_data_.num_speakers, "n_speakers");
    SHERPA_ONNX_READ_META_DATA(meta_data_.has_espeak, "has_espeak");
    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice",
                                                "en-us");
    SHERPA_ONNX_READ_META_DATA_VEC(style_dim_, "style_dim");

    if (style_dim_.size() != 3) {
      SHERPA_ONNX_LOGE("style_dim should be 3-d, given: %d",
                       static_cast<int32_t>(style_dim_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    if (style_dim_[1] != 1) {
      SHERPA_ONNX_LOGE("style_dim[1] should be 1, given: %d", style_dim_[1]);
      SHERPA_ONNX_EXIT(-1);
    }

    meta_data_.max_token_len = style_dim_[0];

    int32_t actual_num_floats = voices_data_length / sizeof(float);
    int32_t expected_num_floats =
        style_dim_[0] * style_dim_[2] * meta_data_.num_speakers;

    if (actual_num_floats != expected_num_floats) {
      SHERPA_ONNX_LOGE(
          "Corrupted voices file. Expected #floats: %d, actual: %d",
          expected_num_floats, actual_num_floats);
      SHERPA_ONNX_EXIT(-1);
    }
    styles_ = std::vector<float>(
        reinterpret_cast<const float *>(voices_data),
        reinterpret_cast<const float *>(voices_data) + actual_num_floats);

    auto shape1 = axera_models_[0]->TensorShape("input_ids");
    if (shape1.size() < 2) {
      SHERPA_ONNX_LOGE("Unexpected shape rank for input_ids: %d",
                       static_cast<int32_t>(shape1.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    max_seq_len_ = shape1[1];
  }

  std::vector<float> LoadVoiceEmbedding(int32_t sid, int32_t phoneme_len) {
    int32_t style_dim0 = style_dim_[0];
    int32_t style_dim1 = style_dim_[2];
    phoneme_len = std::max(phoneme_len, 0);

    sid = std::max(sid, 0);
    if (meta_data_.num_speakers > 0) {
      sid = std::min(sid, meta_data_.num_speakers - 1);
    }

    std::vector<float> ref_s(style_dim1);
    if (phoneme_len < style_dim0) {
      const float *p = styles_.data() + sid * style_dim0 * style_dim1 +
                       phoneme_len * style_dim1;
      std::copy(p, p + style_dim1, ref_s.begin());
    } else {
      int32_t idx = style_dim0 / 2;
      const float *p =
          styles_.data() + sid * style_dim0 * style_dim1 + idx * style_dim1;
      std::copy(p, p + style_dim1, ref_s.begin());
    }
    return ref_s;
  }

  bool RunModels(std::vector<int64_t> &input_ids,
                 const std::vector<float> &ref_s, float speed,
                 std::vector<float> &audio) {
    int32_t actual_len = input_ids.size();
    if (actual_len > max_seq_len_) {
      SHERPA_ONNX_LOGE(
          "Input ids length %d exceeds max_seq_len %d, truncating.", actual_len,
          max_seq_len_);
      input_ids.resize(max_seq_len_);
      actual_len = max_seq_len_;
    }

    int32_t padding_len = max_seq_len_ - actual_len;
    if (padding_len > 0) {
      std::vector<int64_t> padding(padding_len, 0);
      input_ids.insert(input_ids.end(), padding.begin(), padding.end());
    }

    int32_t actual_content_frames = 0;
    int32_t total_frames = 0;
    if (!InferenceSingleChunk(input_ids, ref_s, actual_len, speed, audio,
                              actual_content_frames, total_frames)) {
      return false;
    }

    TrimAudioByContent(audio, actual_content_frames, total_frames, actual_len);

    return true;
  }

  bool InferenceSingleChunk(std::vector<int64_t> &input_ids,
                            const std::vector<float> &ref_s, int32_t actual_len,
                            float speed, std::vector<float> &audio,
                            int32_t &actual_content_frames,
                            int32_t &total_frames) {
    bool is_doubled = false;
    int32_t original_actual_len = actual_len;

    PrepareInputIds(input_ids, actual_len, is_doubled);

    std::vector<int32_t> input_ids_i32(input_ids.begin(), input_ids.end());

    std::vector<int32_t> text_mask;
    ComputeExternalPreprocessing(actual_len, text_mask);

    std::vector<uint8_t> text_mask_u8(text_mask.begin(), text_mask.end());

    if (!axera_models_[0]->SetInputTensorData("input_ids", input_ids_i32.data(),
                                              input_ids_i32.size()) ||
        !axera_models_[0]->SetInputTensorData("ref_s", ref_s.data(),
                                              ref_s.size()) ||
        !axera_models_[0]->SetInputTensorData("text_mask", text_mask_u8.data(),
                                              text_mask_u8.size())) {
      return false;
    }

    if (!axera_models_[0]->Run()) {
      return false;
    }

    std::vector<float> duration =
        axera_models_[0]->GetOutputTensorData("duration");
    std::vector<float> d = axera_models_[0]->GetOutputTensorData("d");

    std::vector<int32_t> pred_dur;
    ProcessDuration(duration, actual_len, speed, pred_dur, total_frames);

    std::vector<float> pred_aln_trg =
        CreateAlignmentMatrix(pred_dur, total_frames);

    int32_t d_shape_2 = 640;
    std::vector<float> en(total_frames * d_shape_2, 0.0f);

    for (int32_t j = 0; j < d_shape_2; ++j) {
      for (int32_t k = 0; k < total_frames; ++k) {
        float sum = 0.0f;
        for (int32_t i = 0; i < max_seq_len_; ++i) {
          sum += d[i * d_shape_2 + j] * pred_aln_trg[i * total_frames + k];
        }
        en[j * total_frames + k] = sum;
      }
    }

    std::vector<float> text_mask_float(text_mask.begin(), text_mask.end());

    if (!axera_models_[1]->SetInputTensorData("en", en.data(), en.size()) ||
        !axera_models_[1]->SetInputTensorData("ref_s", ref_s.data(),
                                              ref_s.size()) ||
        !axera_models_[1]->SetInputTensorData("input_ids", input_ids_i32.data(),
                                              input_ids_i32.size()) ||
        !axera_models_[1]->SetInputTensorData(
            "text_mask", text_mask_float.data(), text_mask_float.size()) ||
        !axera_models_[1]->SetInputTensorData(
            "pred_aln_trg", pred_aln_trg.data(), pred_aln_trg.size())) {
      return false;
    }

    if (!axera_models_[1]->Run()) {
      return false;
    }

    std::vector<float> f0_pred =
        axera_models_[1]->GetOutputTensorData("F0_pred");
    std::vector<float> n_pred = axera_models_[1]->GetOutputTensorData("N_pred");
    std::vector<float> asr = axera_models_[1]->GetOutputTensorData("asr");

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> f0_shape = {1, static_cast<int64_t>(f0_pred.size())};
    Ort::Value f0_tensor =
        Ort::Value::CreateTensor(memory_info, f0_pred.data(), f0_pred.size(),
                                 f0_shape.data(), f0_shape.size());

    auto out4 = model4_->Run({}, model4_input_names_ptr_.data(), &f0_tensor, 1,
                             model4_output_names_ptr_.data(),
                             model4_output_names_ptr_.size());
    float *p_har = out4[0].GetTensorMutableData<float>();
    auto har_shape = out4[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t har_num_elements = 1;
    for (auto d : har_shape) {
      har_num_elements *= d;
    }
    std::vector<float> har(p_har, p_har + har_num_elements);

    if (!axera_models_[2]->SetInputTensorData("asr", asr.data(), asr.size()) ||
        !axera_models_[2]->SetInputTensorData("F0_pred", f0_pred.data(),
                                              f0_pred.size()) ||
        !axera_models_[2]->SetInputTensorData("N_pred", n_pred.data(),
                                              n_pred.size()) ||
        !axera_models_[2]->SetInputTensorData("ref_s", ref_s.data(),
                                              ref_s.size()) ||
        !axera_models_[2]->SetInputTensorData("har", har.data(), har.size())) {
      return false;
    }

    if (!axera_models_[2]->Run()) {
      return false;
    }

    std::vector<float> x_out = axera_models_[2]->GetOutputTensorData("x");
    PostprocessXToAudio(x_out, 23041, audio);

    if (is_doubled) {
      actual_content_frames = std::accumulate(
          pred_dur.begin(), pred_dur.begin() + original_actual_len, 0);
      audio.erase(audio.begin() + audio.size() / 2, audio.end());
      total_frames = total_frames / 2;
    } else {
      actual_content_frames =
          std::accumulate(pred_dur.begin(), pred_dur.begin() + actual_len, 0);
    }
    return true;
  }

  void ProcessDuration(const std::vector<float> &duration, int32_t actual_len,
                       float speed, std::vector<int32_t> &pred_dur,
                       int32_t &total_frames) {
    std::vector<int32_t> pred_dur_original(actual_len, 0);
    std::vector<float> duration_processed = Sigmoid(duration);

    int32_t dur_shape_2 = 50;
    for (int32_t i = 0; i < actual_len; i++) {
      float sum = 0;
      for (int32_t n = 0; n < dur_shape_2; n++) {
        sum += duration_processed[i * dur_shape_2 + n];
      }
      sum /= speed;
      pred_dur_original[i] =
          static_cast<int32_t>(std::max(1.f, std::round(sum)));
    }

    std::vector<int32_t> pred_dur_padding(max_seq_len_ - actual_len, 0);
    pred_dur = pred_dur_original;
    pred_dur.insert(pred_dur.end(), pred_dur_padding.begin(),
                    pred_dur_padding.end());

    int32_t fixed_total_frames = max_seq_len_ * 2;
    int32_t actual_frames =
        std::accumulate(pred_dur.begin(), pred_dur.begin() + actual_len, 0);
    int32_t diff = fixed_total_frames - actual_frames;

    if (diff < 0) {
      auto indices = Argsort(pred_dur, actual_len, true);
      int32_t decreased = 0;
      for (auto idx : indices) {
        if (pred_dur[idx] > 1 && decreased < std::abs(diff)) {
          pred_dur[idx]--;
          decreased++;
        }
        if (decreased >= std::abs(diff)) break;
      }
    }

    actual_frames =
        std::accumulate(pred_dur.begin(), pred_dur.begin() + actual_len, 0);
    int32_t remaining_frames = fixed_total_frames - actual_frames;
    int32_t padding_len = max_seq_len_ - actual_len;

    if (remaining_frames > 0 && padding_len > 0) {
      int32_t frames_per_padding = remaining_frames / padding_len;
      int32_t remainder = remaining_frames % padding_len;

      for (int32_t i = actual_len; i < static_cast<int32_t>(pred_dur.size());
           i++)
        pred_dur[i] = frames_per_padding;

      if (remainder > 0) {
        for (int32_t i = actual_len; i < actual_len + remainder; i++)
          pred_dur[i] += 1;
      }
    }

    total_frames = std::accumulate(pred_dur.begin(), pred_dur.end(), 0);
  }

  std::vector<float> CreateAlignmentMatrix(const std::vector<int32_t> &pred_dur,
                                           int32_t total_frames) {
    std::vector<int32_t> seq_range(max_seq_len_);
    std::iota(seq_range.begin(), seq_range.end(), 0);
    auto indices = NpRepeat(seq_range, pred_dur);

    std::vector<float> pred_aln_trg(max_seq_len_ * total_frames, 0.0f);
    if (!indices.empty()) {
      int32_t col = 0;
      for (auto i : indices) {
        pred_aln_trg[i * total_frames + col] = 1.0f;
        col++;
      }
    }

    return pred_aln_trg;
  }

  void PostprocessXToAudio(const std::vector<float> &x, int32_t num_frames,
                           std::vector<float> &audio) {
    int32_t n_fft = 20;
    int32_t hop_length = 5;
    int32_t half_n_fft = n_fft / 2 + 1;

    std::vector<float> spec_part(half_n_fft * num_frames);
    std::vector<float> phase_part(half_n_fft * num_frames);
    std::vector<float> cos_part(half_n_fft * num_frames);
    spec_part.assign(x.begin(), x.begin() + half_n_fft * num_frames);
    phase_part.assign(x.begin() + half_n_fft * num_frames, x.end());

    for (int32_t i = 0; i < half_n_fft * num_frames; i++) {
      spec_part[i] = std::exp(spec_part[i]);
      phase_part[i] = std::sin(phase_part[i]);
      cos_part[i] = std::sqrt(
          1.f - std::max(0.f, std::min(phase_part[i] * phase_part[i], 1.0f)));
    }

    knf::StftResult stft_result;
    stft_result.num_frames = num_frames;
    stft_result.real.resize(half_n_fft * num_frames);
    stft_result.imag.resize(half_n_fft * num_frames);

    for (int32_t i = 0; i < half_n_fft; i++) {
      for (int32_t n = 0; n < num_frames; n++) {
        float spec = spec_part[i * num_frames + n];
        float real_part = spec * cos_part[i * num_frames + n];
        float imag_part = spec * phase_part[i * num_frames + n];

        // StftResult layout is [frame_index * half_n_fft + bin]
        stft_result.real[n * half_n_fft + i] = real_part;
        stft_result.imag[n * half_n_fft + i] = imag_part;
      }
    }

    knf::StftConfig stft_config;
    stft_config.n_fft = n_fft;
    stft_config.hop_length = hop_length;
    stft_config.win_length = n_fft;
    stft_config.center = true;
    stft_config.normalized = false;
    stft_config.window_type = "hann";
    stft_config.pad_mode = "reflect";

    knf::IStft istft(stft_config);
    audio = istft.Compute(stft_result);
  }

  void TrimAudioByContent(std::vector<float> &audio,
                          int32_t actual_content_frames, int32_t total_frames,
                          int32_t actual_len) {
    int32_t padding_len = max_seq_len_ - actual_len;
    if (padding_len > 0) {
      float content_ratio = actual_content_frames * 1.0f / total_frames;
      int32_t audio_len_to_keep =
          static_cast<int32_t>(audio.size() * content_ratio);
      audio.resize(audio_len_to_keep);
    }
  }

  void PrepareInputIds(std::vector<int64_t> &input_ids, int32_t &actual_len,
                       bool &is_doubled) {
    is_doubled = false;
    int32_t original_actual_len = actual_len;

    if (actual_len <= 32) {  // DOUBLE_INPUT_THRESHOLD
      is_doubled = true;
      std::vector<int64_t> valid_content(input_ids.begin(),
                                         input_ids.begin() + actual_len);
      std::vector<int64_t> input_ids_doubled;
      input_ids_doubled.insert(input_ids_doubled.end(), valid_content.begin(),
                               valid_content.end());
      input_ids_doubled.insert(input_ids_doubled.end(), valid_content.begin(),
                               valid_content.end());

      int32_t padding_len = max_seq_len_ - 2 * actual_len;
      if (padding_len > 0) {
        std::vector<int64_t> padding(padding_len, 0);
        input_ids_doubled.insert(input_ids_doubled.end(), padding.begin(),
                                 padding.end());
      } else {
        input_ids_doubled.resize(max_seq_len_);
      }

      input_ids = input_ids_doubled;
      actual_len = std::min(original_actual_len * 2, max_seq_len_);
    }
  }

  void ComputeExternalPreprocessing(int32_t actual_len,
                                    std::vector<int32_t> &text_mask) {
    text_mask.resize(max_seq_len_);
    for (int32_t i = 0; i < max_seq_len_; i++) {
      text_mask[i] = (i >= actual_len) ? 1 : 0;
    }
  }

 private:
  std::mutex mutex_;
  AxEngineGuard ax_engine_guard_;

  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<std::unique_ptr<AxeraModel>> axera_models_;

  std::unique_ptr<Ort::Session> model4_;
  std::vector<std::string> model4_input_names_;
  std::vector<const char *> model4_input_names_ptr_;
  std::vector<std::string> model4_output_names_;
  std::vector<const char *> model4_output_names_ptr_;

  OfflineTtsKokoroModelMetaData meta_data_;
  std::vector<int32_t> style_dim_;
  std::vector<float> styles_;

  int32_t max_seq_len_;
};

OfflineTtsKokoroModelAxera::~OfflineTtsKokoroModelAxera() = default;

OfflineTtsKokoroModelAxera::OfflineTtsKokoroModelAxera(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsKokoroModelAxera::OfflineTtsKokoroModelAxera(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineTtsKokoroModelAxera::Run(
    const std::vector<int64_t> &x, int64_t sid, float speed) const {
  return impl_->Run(x, sid, speed);
}

const OfflineTtsKokoroModelMetaData &OfflineTtsKokoroModelAxera::GetMetaData()
    const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineTtsKokoroModelAxera::OfflineTtsKokoroModelAxera(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsKokoroModelAxera::OfflineTtsKokoroModelAxera(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
