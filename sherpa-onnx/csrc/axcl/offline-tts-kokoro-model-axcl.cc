// sherpa-onnx/csrc/axcl/offline-tts-kokoro-model-axcl.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/axcl/offline-tts-kokoro-model-axcl.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
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

#include "sherpa-onnx/csrc/axcl/axcl-model.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

constexpr int32_t kKokoroSampleRate = 24000;
constexpr int32_t kKokoroVersion = 2;
constexpr int32_t kKokoroVoiceStyleLength = 510;
constexpr int32_t kKokoroStyleRank = 3;
constexpr int32_t kKokoroStyleMiddleDim = 1;
constexpr int32_t kKokoroTimbreDim = 128;
constexpr int32_t kHarmonicCount = 8;
constexpr int32_t kHarmonicDim = kHarmonicCount + 1;
constexpr int32_t kStftNfft = 20;
constexpr int32_t kStftHop = 5;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kSineAmp = 0.1f;
constexpr float kNoiseStd = 0.003f;
constexpr float kUnvoicedNoiseScale = kSineAmp / 3.0f;
constexpr float kVoicedThreshold = 10.0f;
constexpr char kDefaultVoice[] = "en-us";
constexpr std::array<float, kHarmonicDim> kHarmonicMergeWeights = {
    -0.08154187f, -0.18519667f, -0.18263398f, -0.17837206f, -0.09873895f,
    0.08264039f,  0.08743999f,  -0.39068547f, -0.54774433f,
};
constexpr float kHarmonicMergeBias = -0.02945026f;

static int32_t GetNumElements(const std::vector<int32_t> &shape) {
  if (shape.empty()) {
    return 0;
  }

  int32_t ans = 1;
  for (auto dim : shape) {
    ans *= dim;
  }

  return ans;
}

static int64_t GetNumElements(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return 0;
  }

  int64_t ans = 1;
  for (auto dim : shape) {
    ans *= dim;
  }

  return ans;
}

static std::string Basename(const std::string &path) {
  auto pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  }

  return path.substr(pos + 1);
}

static int32_t ExtractIntegerAfterTag(const std::string &s,
                                      const std::string &tag) {
  auto pos = s.find(tag);
  if (pos == std::string::npos) {
    return -1;
  }

  pos += tag.size();
  int32_t value = 0;
  bool found = false;
  while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
    found = true;
    value = value * 10 + (s[pos] - '0');
    ++pos;
  }

  return found ? value : -1;
}

static std::vector<float> CropLastDim(const std::vector<float> &src,
                                      int32_t outer_dim,
                                      int32_t src_last_dim,
                                      int32_t dst_last_dim) {
  std::vector<float> ans(outer_dim * dst_last_dim);
  for (int32_t i = 0; i != outer_dim; ++i) {
    std::memcpy(ans.data() + i * dst_last_dim,
                src.data() + i * src_last_dim,
                dst_last_dim * sizeof(float));
  }

  return ans;
}

static std::vector<float> PadLastDim(const std::vector<float> &src,
                                     int32_t outer_dim, int32_t src_last_dim,
                                     int32_t dst_last_dim) {
  std::vector<float> ans(outer_dim * dst_last_dim, 0.0f);
  for (int32_t i = 0; i != outer_dim; ++i) {
    std::memcpy(ans.data() + i * dst_last_dim,
                src.data() + i * src_last_dim,
                src_last_dim * sizeof(float));
  }

  return ans;
}

static std::vector<float> PadAlignment(const std::vector<float> &src,
                                       int32_t src_token_len,
                                       int32_t src_frame_len,
                                       int32_t dst_token_len,
                                       int32_t dst_frame_len) {
  std::vector<float> ans(dst_token_len * dst_frame_len, 0.0f);
  for (int32_t i = 0; i != src_token_len; ++i) {
    std::memcpy(ans.data() + i * dst_frame_len,
                src.data() + i * src_frame_len,
                src_frame_len * sizeof(float));
  }

  return ans;
}

static std::vector<uint8_t> BuildPaddedTextMask(int32_t token_len,
                                                int32_t token_bucket) {
  std::vector<uint8_t> mask(token_bucket, 0);
  for (int32_t i = token_len; i < token_bucket; ++i) {
    mask[i] = 1;
  }

  return mask;
}

static std::vector<int64_t> GetOrtTensorShape(Ort::Session *sess,
                                              const std::string &name,
                                              bool is_input) {
  Ort::AllocatorWithDefaultOptions allocator;
  size_t count = is_input ? sess->GetInputCount() : sess->GetOutputCount();
  for (size_t i = 0; i != count; ++i) {
    Ort::AllocatedStringPtr tensor_name =
        is_input ? sess->GetInputNameAllocated(i, allocator)
                 : sess->GetOutputNameAllocated(i, allocator);

    if (name == tensor_name.get()) {
      auto type_info =
          is_input ? sess->GetInputTypeInfo(i) : sess->GetOutputTypeInfo(i);
      return type_info.GetTensorTypeAndShapeInfo().GetShape();
    }
  }

  SHERPA_ONNX_LOGE("Found no %s tensor with name: '%s'",
                   is_input ? "input" : "output", name.c_str());
  return {};
}

struct ModelPaths {
  std::string encoder;
  std::string duration_predictor;
  std::string text_encoder;
  std::string f0n_shared;
  std::string f0n_head;
  std::string decoder_front;
  std::string vocoder_core;
  std::string vocoder_tail;
};

static ModelPaths ParseModelPaths(const std::string &model_str) {
  std::vector<std::string> files;
  SplitStringToVector(model_str, ",", false, &files);

  ModelPaths paths;
  for (const auto &f : files) {
    if (!FileExists(f)) {
      SHERPA_ONNX_LOGE("Model file '%s' does not exist", f.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    std::string name = Basename(f);
    if (name.find("duration_predictor") != std::string::npos) {
      paths.duration_predictor = f;
    } else if (name.find("text_encoder_token_") != std::string::npos) {
      paths.text_encoder = f;
    } else if (name.find("encoder_token_") != std::string::npos) {
      paths.encoder = f;
    } else if (name.find("f0n_shared_frame_") != std::string::npos) {
      paths.f0n_shared = f;
    } else if (name.find("f0n_head_frame_") != std::string::npos) {
      paths.f0n_head = f;
    } else if (name.find("decoder_front") != std::string::npos) {
      paths.decoder_front = f;
    } else if (name.find("vocoder_tail") != std::string::npos) {
      paths.vocoder_tail = f;
    } else if (name.find("vocoder_core") != std::string::npos) {
      paths.vocoder_core = f;
    }
  }

  if (paths.encoder.empty() || paths.duration_predictor.empty() ||
      paths.text_encoder.empty() || paths.f0n_shared.empty() ||
      paths.f0n_head.empty() || paths.decoder_front.empty() ||
      paths.vocoder_core.empty() || paths.vocoder_tail.empty()) {
    SHERPA_ONNX_LOGE(
        "For the new Axcl Kokoro pipeline, --kokoro-model must contain these "
        "8 files in any order: encoder_token_*.axmodel, "
        "duration_predictor.onnx, text_encoder_token_*_frame_*.axmodel, "
        "f0n_shared_frame_*.axmodel, f0n_head_frame_*.axmodel, "
        "decoder_front*.axmodel, vocoder_core*.axmodel, "
        "vocoder_tail*.onnx. Given: %s",
        model_str.c_str());
    SHERPA_ONNX_EXIT(-1);
  }

  return paths;
}

}  // namespace

class OfflineTtsKokoroModelAxcl::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(config, [](const std::string &filename) { return ReadFile(filename); });
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    Init(config, [mgr](const std::string &filename) { return ReadFile(mgr, filename); });
  }

  const OfflineTtsKokoroModelMetaData &GetMetaData() const {
    return meta_data_;
  }

  std::vector<float> Run(const std::vector<int64_t> &input_ids, int64_t sid,
                         float speed) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (input_ids.size() < 2) {
      return {};
    }

    if (config_.kokoro.length_scale != 1 && speed == 1) {
      speed = 1.0f / config_.kokoro.length_scale;
    }

    int32_t phoneme_len = static_cast<int32_t>(input_ids.size()) - 2;
    if (phoneme_len < 0) {
      phoneme_len = 0;
    }

    std::vector<float> ref_s =
        LoadVoiceEmbedding(static_cast<int32_t>(sid), phoneme_len);

    std::vector<float> audio;
    if (!RunModels(input_ids, ref_s, speed, &audio)) {
      SHERPA_ONNX_LOGE("Run Kokoro Axcl static pipeline failed");
      return {};
    }

    return audio;
  }

 private:
  template <typename ReadBytes>
  void Init(const OfflineTtsModelConfig &config, ReadBytes &&read_bytes) {
    if (config_.provider != "axcl") {
      SHERPA_ONNX_LOGE(
          "This model only supports axcl provider. Please use provider=axcl");
      SHERPA_ONNX_EXIT(-1);
    }

    auto paths = ParseModelPaths(config.kokoro.model);

    auto encoder_buf = read_bytes(paths.encoder);
    encoder_model_ = std::make_unique<AxclModel>(encoder_buf.data(),
                                                 encoder_buf.size());

    auto text_encoder_buf = read_bytes(paths.text_encoder);
    text_encoder_model_ = std::make_unique<AxclModel>(
        text_encoder_buf.data(), text_encoder_buf.size());

    auto f0n_shared_buf = read_bytes(paths.f0n_shared);
    f0n_shared_model_ =
        std::make_unique<AxclModel>(f0n_shared_buf.data(), f0n_shared_buf.size());

    auto f0n_head_buf = read_bytes(paths.f0n_head);
    f0n_head_model_ =
        std::make_unique<AxclModel>(f0n_head_buf.data(), f0n_head_buf.size());

    auto decoder_front_buf = read_bytes(paths.decoder_front);
    decoder_front_model_ = std::make_unique<AxclModel>(decoder_front_buf.data(),
                                                       decoder_front_buf.size());

    auto vocoder_core_buf = read_bytes(paths.vocoder_core);
    vocoder_core_model_ = std::make_unique<AxclModel>(vocoder_core_buf.data(),
                              vocoder_core_buf.size());

    auto vocoder_tail_buf = read_bytes(paths.vocoder_tail);
    vocoder_tail_ = std::make_unique<Ort::Session>(
      env_, vocoder_tail_buf.data(), vocoder_tail_buf.size(), sess_opts_);
    GetInputNames(vocoder_tail_.get(), &vocoder_tail_input_names_,
            &vocoder_tail_input_names_ptr_);
    GetOutputNames(vocoder_tail_.get(), &vocoder_tail_output_names_,
             &vocoder_tail_output_names_ptr_);

    auto duration_buf = read_bytes(paths.duration_predictor);
    duration_predictor_ = std::make_unique<Ort::Session>(
        env_, duration_buf.data(), duration_buf.size(), sess_opts_);
    GetInputNames(duration_predictor_.get(), &duration_input_names_,
                  &duration_input_names_ptr_);
    GetOutputNames(duration_predictor_.get(), &duration_output_names_,
                   &duration_output_names_ptr_);

    InitModelShapes(paths);

    auto voices_buf = read_bytes(config.kokoro.voices);
    LoadVoices(voices_buf.data(), voices_buf.size());
  }

  void InitModelShapes(const ModelPaths &paths) {
    auto encoder_input_shape = encoder_model_->TensorShape("input_ids");
    auto encoder_mask_shape = encoder_model_->TensorShape("text_mask");
    auto text_encoder_input_shape = text_encoder_model_->TensorShape("input_ids");
    auto text_encoder_aln_shape = text_encoder_model_->TensorShape("pred_aln_trg");
    auto f0n_shared_input_shape = f0n_shared_model_->TensorShape("en");
    auto f0n_head_ref_shape = f0n_head_model_->TensorShape("ref_s");
    auto f0n_head_output_shape = f0n_head_model_->TensorShape("F0_pred");
    auto decoder_asr_shape = decoder_front_model_->TensorShape("asr");
    auto decoder_timbre_shape = decoder_front_model_->TensorShape("timbre");
    auto vocoder_core_state_shape =
      vocoder_core_model_->TensorShape("decoder_state");
    auto vocoder_core_timbre_shape = vocoder_core_model_->TensorShape("timbre");
    auto vocoder_core_har_shape = vocoder_core_model_->TensorShape("har");
    auto vocoder_core_hidden_shape =
      vocoder_core_model_->TensorShape("vocoder_hidden");
    auto vocoder_tail_hidden_shape =
      GetOrtTensorShape(vocoder_tail_.get(), "vocoder_hidden", true);
    auto vocoder_tail_output_shape =
      GetOrtTensorShape(vocoder_tail_.get(), "waveform", false);

    if (encoder_input_shape.size() != 2 || encoder_mask_shape.size() != 2 ||
        text_encoder_input_shape.size() != 2 || text_encoder_aln_shape.size() != 3 ||
        f0n_shared_input_shape.size() != 3 || f0n_head_ref_shape.size() != 2 ||
        f0n_head_output_shape.size() != 2 || decoder_asr_shape.size() != 3 ||
      decoder_timbre_shape.size() != 2 || vocoder_core_state_shape.size() != 3 ||
      vocoder_core_timbre_shape.size() != 2 ||
      vocoder_core_har_shape.size() != 3 ||
      vocoder_core_hidden_shape.size() != 3 ||
      vocoder_tail_hidden_shape.size() != 3 ||
      vocoder_tail_output_shape.size() != 1) {
      SHERPA_ONNX_LOGE("Unexpected Kokoro Axcl model shapes in: %s",
                       config_.kokoro.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    token_bucket_ = encoder_input_shape[1];
    frame_bucket_ = text_encoder_aln_shape[2];
    pitch_bucket_ = f0n_head_output_shape[1];
    ref_s_dim_ = f0n_head_ref_shape[1];
    timbre_dim_ = decoder_timbre_shape[1];
    har_channels_ = vocoder_core_har_shape[1];
    har_frames_ = vocoder_core_har_shape[2];
    vocoder_hidden_channels_ = vocoder_core_hidden_shape[1];
    vocoder_hidden_frames_ = vocoder_core_hidden_shape[2];
    sample_bucket_ = static_cast<int32_t>(GetNumElements(vocoder_tail_output_shape));

    if (token_bucket_ != text_encoder_input_shape[1] ||
        token_bucket_ != text_encoder_aln_shape[1] ||
        frame_bucket_ != f0n_shared_input_shape[2] ||
        frame_bucket_ != decoder_asr_shape[2]) {
      SHERPA_ONNX_LOGE("Kokoro Axcl bucket mismatch detected in: %s",
                       config_.kokoro.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (ref_s_dim_ <= 0 || timbre_dim_ <= 0 || timbre_dim_ > ref_s_dim_) {
      SHERPA_ONNX_LOGE("Invalid Kokoro style dims. ref_s=%d, timbre=%d",
                       ref_s_dim_, timbre_dim_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (vocoder_core_state_shape != decoder_front_model_->TensorShape("decoder_state") ||
        vocoder_core_timbre_shape[1] != timbre_dim_ ||
        vocoder_tail_hidden_shape[1] != vocoder_hidden_channels_ ||
        vocoder_tail_hidden_shape[2] != vocoder_hidden_frames_) {
      SHERPA_ONNX_LOGE("Kokoro Axcl vocoder_core/tail shape mismatch in: %s",
                       config_.kokoro.model.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (har_channels_ != kStftNfft + 2) {
      SHERPA_ONNX_LOGE("Expected har channels %d. Given: %d", kStftNfft + 2,
                       har_channels_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (sample_bucket_ % pitch_bucket_ != 0 || sample_bucket_ % frame_bucket_ != 0) {
      SHERPA_ONNX_LOGE(
          "Unexpected Kokoro Axcl bucket relation. sample=%d pitch=%d frame=%d",
          sample_bucket_, pitch_bucket_, frame_bucket_);
      SHERPA_ONNX_EXIT(-1);
    }

    upsample_scale_ = sample_bucket_ / pitch_bucket_;
    samples_per_frame_ = sample_bucket_ / frame_bucket_;
    int32_t expected_har_frames = sample_bucket_ / kStftHop + 1;
    if (har_frames_ != expected_har_frames) {
      SHERPA_ONNX_LOGE("Expected har frames %d. Given: %d", expected_har_frames,
                       har_frames_);
      SHERPA_ONNX_EXIT(-1);
    }

    if (ExtractIntegerAfterTag(Basename(paths.encoder), "encoder_token_") != -1 &&
        ExtractIntegerAfterTag(Basename(paths.encoder), "encoder_token_") !=
            token_bucket_) {
      SHERPA_ONNX_LOGE("Encoder filename bucket does not match model shape: %s",
                       paths.encoder.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void LoadVoices(const char *voices_data, size_t voices_data_length) {
    meta_data_.sample_rate = kKokoroSampleRate;
    meta_data_.version = kKokoroVersion;
    meta_data_.has_espeak = 1;
    meta_data_.voice = config_.kokoro.lang.empty() ? kDefaultVoice : config_.kokoro.lang;
    meta_data_.max_token_len = kKokoroVoiceStyleLength;

    style_dim_ = {kKokoroVoiceStyleLength, kKokoroStyleMiddleDim, ref_s_dim_};
    if (style_dim_.size() != kKokoroStyleRank || style_dim_[1] != 1) {
      SHERPA_ONNX_LOGE("Unexpected Kokoro style_dim");
      SHERPA_ONNX_EXIT(-1);
    }

    int32_t actual_num_floats = static_cast<int32_t>(voices_data_length / sizeof(float));
    int32_t floats_per_speaker = style_dim_[0] * style_dim_[2];
    if (floats_per_speaker <= 0 || actual_num_floats % floats_per_speaker != 0) {
      SHERPA_ONNX_LOGE(
          "Corrupted voices file '%s'. Expected a multiple of %d floats. Got %d",
          config_.kokoro.voices.c_str(), floats_per_speaker, actual_num_floats);
      SHERPA_ONNX_EXIT(-1);
    }

    meta_data_.num_speakers = actual_num_floats / floats_per_speaker;
    styles_ = std::vector<float>(reinterpret_cast<const float *>(voices_data),
                                 reinterpret_cast<const float *>(voices_data) +
                                     actual_num_floats);
  }

  std::vector<float> LoadVoiceEmbedding(int32_t sid, int32_t phoneme_len) const {
    int32_t style_len = style_dim_[0];
    int32_t style_dim = style_dim_[2];

    sid = std::max(sid, 0);
    if (meta_data_.num_speakers > 0) {
      sid = std::min(sid, meta_data_.num_speakers - 1);
    }

    phoneme_len = std::max(phoneme_len, 0);

    std::vector<float> ref_s(style_dim);
    int32_t index = phoneme_len < style_len ? phoneme_len : style_len / 2;
    const float *src = styles_.data() + sid * style_len * style_dim +
                       index * style_dim;
    std::copy(src, src + style_dim, ref_s.begin());
    return ref_s;
  }

  std::vector<float> BuildHar(const std::vector<float> &f0_pred) const {
    std::vector<float> phase_coarse(pitch_bucket_ * kHarmonicDim, 0.0f);
    std::array<float, kHarmonicDim> cumulative{};

    for (int32_t t = 0; t != pitch_bucket_; ++t) {
      float base_f0 = f0_pred[t];
      for (int32_t h = 0; h != kHarmonicDim; ++h) {
        float harmonic = base_f0 * static_cast<float>(h + 1);
        float rad = std::fmod(harmonic / static_cast<float>(kKokoroSampleRate), 1.0f);
        if (rad < 0) {
          rad += 1.0f;
        }
        cumulative[h] += rad;
        phase_coarse[t * kHarmonicDim + h] =
            cumulative[h] * kTwoPi * static_cast<float>(upsample_scale_);
      }
    }

    std::vector<float> sine_merge(sample_bucket_);
    std::mt19937 rng(0);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    for (int32_t sample_index = 0; sample_index != sample_bucket_; ++sample_index) {
      int32_t coarse_index = sample_index / upsample_scale_;
      float coarse_f0 = f0_pred[coarse_index];
      float uv = coarse_f0 > kVoicedThreshold ? 1.0f : 0.0f;
      float noise_amp = uv * kNoiseStd + (1.0f - uv) * kUnvoicedNoiseScale;

      float src = (static_cast<float>(sample_index) + 0.5f) /
                      static_cast<float>(upsample_scale_) -
                  0.5f;
      if (src < 0.0f) {
        src = 0.0f;
      }

      int32_t left = static_cast<int32_t>(std::floor(src));
      int32_t right = std::min(left + 1, pitch_bucket_ - 1);
      float w = src - static_cast<float>(left);
      if (left >= pitch_bucket_ - 1) {
        left = pitch_bucket_ - 1;
        right = left;
        w = 0.0f;
      }

      float merged = kHarmonicMergeBias;
      for (int32_t h = 0; h != kHarmonicDim; ++h) {
        float phase_left = phase_coarse[left * kHarmonicDim + h];
        float phase_right = phase_coarse[right * kHarmonicDim + h];
        float phase = phase_left * (1.0f - w) + phase_right * w;
        float sine = std::sin(phase) * kSineAmp;
        float noise = noise_amp * normal(rng);
        float sine_wave = sine * uv + noise;
        merged += sine_wave * kHarmonicMergeWeights[h];
      }

      sine_merge[sample_index] = std::tanh(merged);
    }

    std::vector<float> padded(sample_bucket_ + kStftNfft, 0.0f);
    int32_t pad = kStftNfft / 2;
    std::copy(sine_merge.begin(), sine_merge.end(), padded.begin() + pad);
    std::fill(padded.begin(), padded.begin() + pad, sine_merge.front());
    std::fill(padded.end() - pad, padded.end(), sine_merge.back());

    std::array<float, kStftNfft> window{};
    std::array<std::array<float, kStftNfft>, kStftNfft / 2 + 1> cos_table{};
    std::array<std::array<float, kStftNfft>, kStftNfft / 2 + 1> sin_table{};

    for (int32_t n = 0; n != kStftNfft; ++n) {
      window[n] = 0.5f - 0.5f * std::cos(kTwoPi * n / static_cast<float>(kStftNfft));
    }

    for (int32_t k = 0; k != kStftNfft / 2 + 1; ++k) {
      for (int32_t n = 0; n != kStftNfft; ++n) {
        float angle = kTwoPi * k * n / static_cast<float>(kStftNfft);
        cos_table[k][n] = std::cos(angle);
        sin_table[k][n] = -std::sin(angle);
      }
    }

    std::vector<float> har(har_channels_ * har_frames_);
    for (int32_t frame = 0; frame != har_frames_; ++frame) {
      const float *segment = padded.data() + frame * kStftHop;
      for (int32_t k = 0; k != kStftNfft / 2 + 1; ++k) {
        float real = 0.0f;
        float imag = 0.0f;
        for (int32_t n = 0; n != kStftNfft; ++n) {
          float value = segment[n] * window[n];
          real += value * cos_table[k][n];
          imag += value * sin_table[k][n];
        }

        har[k * har_frames_ + frame] =
            std::sqrt(real * real + imag * imag + 1e-14f);
        float phase = std::atan2(imag, real);
        if (imag == 0.0f && real < 0.0f) {
          phase = kPi;
        }
        har[(k + kStftNfft / 2 + 1) * har_frames_ + frame] = phase;
      }
    }

    return har;
  }

  bool RunModels(const std::vector<int64_t> &input_ids,
                 const std::vector<float> &ref_s, float speed,
                 std::vector<float> *audio) {
    int32_t token_length = static_cast<int32_t>(input_ids.size());
    if (token_length > token_bucket_) {
      SHERPA_ONNX_LOGE(
          "Token length %d exceeds Axcl token bucket %d. Use a shorter text "
          "or export a larger static bucket.",
          token_length, token_bucket_);
      return false;
    }

    std::vector<int32_t> padded_input_ids(token_bucket_, 0);
    for (int32_t i = 0; i != token_length; ++i) {
      padded_input_ids[i] = static_cast<int32_t>(input_ids[i]);
    }

    std::vector<uint8_t> padded_text_mask =
        BuildPaddedTextMask(token_length, token_bucket_);

    if (!encoder_model_->SetInputTensorData("input_ids", padded_input_ids.data(),
                                            padded_input_ids.size()) ||
        !encoder_model_->SetInputTensorData("text_mask", padded_text_mask.data(),
                                            padded_text_mask.size()) ||
        !encoder_model_->Run()) {
      return false;
    }

    std::vector<float> padded_d_en = encoder_model_->GetOutputTensorData("d_en");
    std::vector<float> d_en =
        CropLastDim(padded_d_en, 512, token_bucket_, token_length);

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> d_en_shape = {1, 512, token_length};
    std::array<int64_t, 2> ref_s_shape = {1, ref_s_dim_};
    std::array<int64_t, 1> input_lengths_shape = {1};
    std::array<int64_t, 2> duration_text_mask_shape = {1, token_length};
    std::array<int64_t, 1> speed_shape = {1};

    std::array<int64_t, 1> input_lengths = {token_length};
    std::unique_ptr<bool[]> duration_text_mask(new bool[token_length]());

    Ort::Value d_en_tensor = Ort::Value::CreateTensor(
        memory_info, d_en.data(), d_en.size(), d_en_shape.data(), d_en_shape.size());
    Ort::Value ref_s_tensor = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(ref_s.data()), ref_s.size(),
        ref_s_shape.data(), ref_s_shape.size());
    Ort::Value input_lengths_tensor = Ort::Value::CreateTensor(
        memory_info, input_lengths.data(), input_lengths.size(),
        input_lengths_shape.data(), input_lengths_shape.size());
    Ort::Value duration_text_mask_tensor = Ort::Value::CreateTensor(
        memory_info, duration_text_mask.get(), token_length,
        duration_text_mask_shape.data(), duration_text_mask_shape.size());
    Ort::Value speed_tensor = Ort::Value::CreateTensor(
        memory_info, &speed, 1, speed_shape.data(), speed_shape.size());

    std::array<Ort::Value, 5> duration_inputs = {
        std::move(d_en_tensor), std::move(ref_s_tensor),
        std::move(input_lengths_tensor), std::move(duration_text_mask_tensor),
        std::move(speed_tensor)};

    auto duration_outputs = duration_predictor_->Run(
        {}, duration_input_names_ptr_.data(), duration_inputs.data(),
        duration_inputs.size(), duration_output_names_ptr_.data(),
        duration_output_names_ptr_.size());

    auto pred_aln_shape = duration_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    auto en_shape = duration_outputs[3].GetTensorTypeAndShapeInfo().GetShape();
    if (pred_aln_shape.size() != 3 || en_shape.size() != 3) {
      SHERPA_ONNX_LOGE("Unexpected duration predictor output shapes");
      return false;
    }

    int32_t frame_length = static_cast<int32_t>(pred_aln_shape[2]);
    if (frame_length > frame_bucket_) {
      SHERPA_ONNX_LOGE(
          "Frame length %d exceeds Axcl frame bucket %d. Use a shorter text, "
          "increase speed, or export a larger static bucket.",
          frame_length, frame_bucket_);
      return false;
    }

    const float *pred_aln_ptr = duration_outputs[2].GetTensorData<float>();
    int32_t pred_aln_elems = static_cast<int32_t>(
        duration_outputs[2].GetTensorTypeAndShapeInfo().GetElementCount());
    std::vector<float> pred_aln_trg(pred_aln_ptr, pred_aln_ptr + pred_aln_elems);
    std::vector<float> padded_alignment = PadAlignment(
        pred_aln_trg, token_length, frame_length, token_bucket_, frame_bucket_);

    if (!text_encoder_model_->SetInputTensorData("input_ids", padded_input_ids.data(),
                                                 padded_input_ids.size()) ||
        !text_encoder_model_->SetInputTensorData("pred_aln_trg",
                                                 padded_alignment.data(),
                                                 padded_alignment.size()) ||
        !text_encoder_model_->SetInputTensorData("text_mask",
                                                 padded_text_mask.data(),
                                                 padded_text_mask.size()) ||
        !text_encoder_model_->Run()) {
      return false;
    }

    std::vector<float> asr = text_encoder_model_->GetOutputTensorData("asr");

    const float *en_ptr = duration_outputs[3].GetTensorData<float>();
    int32_t en_elems = static_cast<int32_t>(
        duration_outputs[3].GetTensorTypeAndShapeInfo().GetElementCount());
    std::vector<float> en(en_ptr, en_ptr + en_elems);
    std::vector<float> padded_en = PadLastDim(en, 640, frame_length, frame_bucket_);

    if (!f0n_shared_model_->SetInputTensorData("en", padded_en.data(),
                                               padded_en.size()) ||
        !f0n_shared_model_->Run()) {
      return false;
    }

    std::vector<float> shared = f0n_shared_model_->GetOutputTensorData("shared");
    if (!f0n_head_model_->SetInputTensorData("shared", shared.data(), shared.size()) ||
        !f0n_head_model_->SetInputTensorData("ref_s", ref_s.data(), ref_s.size()) ||
        !f0n_head_model_->Run()) {
      return false;
    }

    std::vector<float> f0_pred = f0n_head_model_->GetOutputTensorData("F0_pred");
    std::vector<float> n_pred = f0n_head_model_->GetOutputTensorData("N_pred");
    std::vector<float> timbre(ref_s.begin(), ref_s.begin() + timbre_dim_);

    if (!decoder_front_model_->SetInputTensorData("asr", asr.data(), asr.size()) ||
        !decoder_front_model_->SetInputTensorData("F0_pred", f0_pred.data(),
                                                  f0_pred.size()) ||
        !decoder_front_model_->SetInputTensorData("N_pred", n_pred.data(),
                                                  n_pred.size()) ||
        !decoder_front_model_->SetInputTensorData("timbre", timbre.data(),
                                                  timbre.size()) ||
        !decoder_front_model_->Run()) {
      return false;
    }

    std::vector<float> decoder_state =
        decoder_front_model_->GetOutputTensorData("decoder_state");
    std::vector<float> har = BuildHar(f0_pred);

    if (!vocoder_core_model_->SetInputTensorData("decoder_state",
                           decoder_state.data(),
                           decoder_state.size()) ||
      !vocoder_core_model_->SetInputTensorData("timbre", timbre.data(),
                           timbre.size()) ||
      !vocoder_core_model_->SetInputTensorData("har", har.data(), har.size()) ||
      !vocoder_core_model_->Run()) {
      return false;
    }

    std::vector<float> vocoder_hidden =
      vocoder_core_model_->GetOutputTensorData("vocoder_hidden");
    std::array<int64_t, 3> vocoder_hidden_shape = {
      1, vocoder_hidden_channels_, vocoder_hidden_frames_};
    Ort::Value vocoder_hidden_tensor = Ort::Value::CreateTensor(
      memory_info, vocoder_hidden.data(), vocoder_hidden.size(),
      vocoder_hidden_shape.data(), vocoder_hidden_shape.size());
    std::array<Ort::Value, 1> vocoder_tail_inputs = {
      std::move(vocoder_hidden_tensor)};

    auto vocoder_tail_outputs = vocoder_tail_->Run(
      {}, vocoder_tail_input_names_ptr_.data(), vocoder_tail_inputs.data(),
      vocoder_tail_inputs.size(), vocoder_tail_output_names_ptr_.data(),
      vocoder_tail_output_names_ptr_.size());
    if (vocoder_tail_outputs.empty()) {
      SHERPA_ONNX_LOGE("Unexpected empty vocoder_tail outputs");
      return false;
    }

    const float *waveform_ptr = vocoder_tail_outputs[0].GetTensorData<float>();
    int32_t waveform_elems = static_cast<int32_t>(
      vocoder_tail_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount());
    *audio = std::vector<float>(waveform_ptr, waveform_ptr + waveform_elems);
    int32_t sample_length = frame_length * samples_per_frame_;
    if (sample_length < static_cast<int32_t>(audio->size())) {
      audio->resize(sample_length);
    }

    return true;
  }

 private:
  mutable std::mutex mutex_;

  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<AxclModel> encoder_model_;
  std::unique_ptr<AxclModel> text_encoder_model_;
  std::unique_ptr<AxclModel> f0n_shared_model_;
  std::unique_ptr<AxclModel> f0n_head_model_;
  std::unique_ptr<AxclModel> decoder_front_model_;
  std::unique_ptr<AxclModel> vocoder_core_model_;

  std::unique_ptr<Ort::Session> duration_predictor_;
  std::unique_ptr<Ort::Session> vocoder_tail_;
  std::vector<std::string> duration_input_names_;
  std::vector<const char *> duration_input_names_ptr_;
  std::vector<std::string> duration_output_names_;
  std::vector<const char *> duration_output_names_ptr_;
  std::vector<std::string> vocoder_tail_input_names_;
  std::vector<const char *> vocoder_tail_input_names_ptr_;
  std::vector<std::string> vocoder_tail_output_names_;
  std::vector<const char *> vocoder_tail_output_names_ptr_;

  OfflineTtsKokoroModelMetaData meta_data_;
  std::vector<int32_t> style_dim_;
  std::vector<float> styles_;

  int32_t token_bucket_ = 0;
  int32_t frame_bucket_ = 0;
  int32_t pitch_bucket_ = 0;
  int32_t sample_bucket_ = 0;
  int32_t ref_s_dim_ = 0;
  int32_t timbre_dim_ = 0;
  int32_t har_channels_ = 0;
  int32_t har_frames_ = 0;
  int64_t vocoder_hidden_channels_ = 0;
  int64_t vocoder_hidden_frames_ = 0;
  int32_t upsample_scale_ = 0;
  int32_t samples_per_frame_ = 0;
};

OfflineTtsKokoroModelAxcl::~OfflineTtsKokoroModelAxcl() = default;

OfflineTtsKokoroModelAxcl::OfflineTtsKokoroModelAxcl(
    const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineTtsKokoroModelAxcl::OfflineTtsKokoroModelAxcl(
    Manager *mgr, const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

std::vector<float> OfflineTtsKokoroModelAxcl::Run(
    const std::vector<int64_t> &x, int64_t sid, float speed) const {
  return impl_->Run(x, sid, speed);
}

const OfflineTtsKokoroModelMetaData &OfflineTtsKokoroModelAxcl::GetMetaData()
    const {
  return impl_->GetMetaData();
}

#if __ANDROID_API__ >= 9
template OfflineTtsKokoroModelAxcl::OfflineTtsKokoroModelAxcl(
    AAssetManager *mgr, const OfflineTtsModelConfig &config);
#endif

#if __OHOS__
template OfflineTtsKokoroModelAxcl::OfflineTtsKokoroModelAxcl(
    NativeResourceManager *mgr, const OfflineTtsModelConfig &config);
#endif

}  // namespace sherpa_onnx
