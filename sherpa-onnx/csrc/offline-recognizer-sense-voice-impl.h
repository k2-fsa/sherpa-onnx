// sherpa-onnx/csrc/offline-recognizer-sense-voice-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#include "sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-sense-voice-model.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

static OfflineRecognitionResult ConvertSenseVoiceResult(
    const OfflineCtcDecoderResult &src, const SymbolTable &sym_table,
    int32_t frame_shift_ms, int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;

  for (int32_t i = 4; i < src.tokens.size(); ++i) {
    auto sym = sym_table[src.tokens[i]];
    text.append(sym);

    r.tokens.push_back(std::move(sym));
  }
  r.text = std::move(text);

  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;

  for (int32_t i = 4; i < src.timestamps.size(); ++i) {
    float time = frame_shift_s * (src.timestamps[i] - 4);
    r.timestamps.push_back(time);
  }

  r.words = std::move(src.words);

  return r;
}

class OfflineRecognizerSenseVoiceImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerSenseVoiceImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineSenseVoiceModel>(config.model_config)) {
    const auto &meta_data = model_->GetModelMetadata();
    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineCtcGreedySearchDecoder>(meta_data.blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    InitFeatConfig();
  }

#if __ANDROID_API__ >= 9
  OfflineRecognizerSenseVoiceImpl(AAssetManager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineSenseVoiceModel>(mgr,
                                                        config.model_config)) {
    const auto &meta_data = model_->GetModelMetadata();
    if (config.decoding_method == "greedy_search") {
      decoder_ =
          std::make_unique<OfflineCtcGreedySearchDecoder>(meta_data.blank_id);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search is supported at present. Given %s",
                       config.decoding_method.c_str());
      exit(-1);
    }

    InitFeatConfig();
  }
#endif

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    if (n == 1) {
      DecodeOneStream(ss[0]);
      return;
    }

    const auto &meta_data = model_->GetModelMetadata();
    // 1. Apply LFR
    // 2. Apply CMVN
    //
    // Please refer to
    // https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45555.pdf
    // for what LFR means
    //
    // "Lower Frame Rate Neural Network Acoustic Models"
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> features;
    features.reserve(n);

    // int32_t feat_dim = config_.feat_config.feature_dim *
    // meta_data.window_size;

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int32_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      std::vector<float> fs = ss[i]->GetFrames();
      SHERPA_ONNX_LOGE("feat:%f,%f,%f", fs[0], fs[1], fs[2]);
      int32_t feat_dim = ss[i]->FeatureDim();
      std::vector<std::vector<float>> feats;
      int32_t num_frames = fs.size() / feat_dim;
      float *p = fs.data();
      for (int32_t i = 0; i != num_frames; ++i) {
        std::vector<float> frame_vector(p, p + feat_dim);
        feats.emplace_back(std::move(frame_vector));
        p += feat_dim;
      }

      LfrCmvn(feats);
      num_frames = feats.size();
      const int feature_dim = feats[0].size();

      std::vector<float> f;
      for (const auto &feat : feats) {
        f.insert(f.end(), feat.begin(), feat.end());
      }
      SHERPA_ONNX_LOGE("LfrCmvn feat:%.8f,%.8f,%.8f", f[0], f[1], f[2]);

      features_vec[i] = std::move(f);
      features_length_vec[i] = num_frames;

      std::array<int64_t, 2> shape = {num_frames, feature_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }

    std::vector<const Ort::Value *> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = &features[i];
    }

    std::array<int64_t, 1> features_length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    // Caution(fangjun): We cannot pad it with log(eps),
    // i.e., -23.025850929940457f
    Ort::Value x = PadSequence(model_->Allocator(), features_pointer, 0);

    int32_t language = 0;
    if (config_.model_config.sense_voice.language.empty()) {
      language = 0;
    } else if (meta_data.lang2id.count(
                   config_.model_config.sense_voice.language)) {
      language =
          meta_data.lang2id.at(config_.model_config.sense_voice.language);
    } else {
      SHERPA_ONNX_LOGE("Unknown language: %s. Use 0 instead.",
                       config_.model_config.sense_voice.language.c_str());
    }

    std::vector<int32_t> language_array(n);
    std::fill(language_array.begin(), language_array.end(), language);

    std::vector<int32_t> text_norm_array(n);
    std::fill(text_norm_array.begin(), text_norm_array.end(),
              config_.model_config.sense_voice.use_itn
                  ? meta_data.with_itn_id
                  : meta_data.without_itn_id);

    Ort::Value language_tensor = Ort::Value::CreateTensor(
        memory_info, language_array.data(), n, features_length_shape.data(),
        features_length_shape.size());

    Ort::Value text_norm_tensor = Ort::Value::CreateTensor(
        memory_info, text_norm_array.data(), n, features_length_shape.data(),
        features_length_shape.size());

    Ort::Value logits{nullptr};
    try {
      logits = model_->Forward(std::move(x), std::move(x_length),
                               std::move(language_tensor),
                               std::move(text_norm_tensor));
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE("\n\nCaught exception:\n\n%s\n\nReturn an empty result",
                       ex.what());
      return;
    }

    // decoder_->Decode() requires that logits_length is of dtype int64
    std::vector<int64_t> features_length_vec_64;
    features_length_vec_64.reserve(n);
    for (auto i : features_length_vec) {
      i += 4;
      features_length_vec_64.push_back(i);
    }

    Ort::Value logits_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec_64.data(), n,
        features_length_shape.data(), features_length_shape.size());

    auto results =
        decoder_->Decode(std::move(logits), std::move(logits_length));

    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = meta_data.window_shift;
    for (int32_t i = 0; i != n; ++i) {
      auto r = ConvertSenseVoiceResult(results[i], symbol_table_,
                                       frame_shift_ms, subsampling_factor);
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void DecodeOneStream(OfflineStream *s) const {
    const auto &meta_data = model_->GetModelMetadata();

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<float> fs = s->GetFrames();
   // SHERPA_ONNX_LOGE("feat:%f,%f,%f", fs[0], fs[1], fs[2]);
    int32_t feat_dim = s->FeatureDim();
    std::vector<std::vector<float>> feats;
    int32_t num_frames = fs.size() / feat_dim;
    float *p = fs.data();
    for (int32_t i = 0; i != num_frames; ++i) {
      std::vector<float> frame_vector(p, p + feat_dim);
      feats.emplace_back(std::move(frame_vector));
      p += feat_dim;
    }
    LfrCmvn(feats);
    num_frames = feats.size();
    const int feature_dim = feats[0].size();

    std::vector<float> f;
    for (const auto &feat : feats) {
      f.insert(f.end(), feat.begin(), feat.end());
    }
    //SHERPA_ONNX_LOGE("LfrCmvn feat:%.8f,%.8f,%.8f, num_frames:%2d", f[0], f[1], f[2], num_frames);
    std::array<int64_t, 3> shape = {1, num_frames, feature_dim};
    Ort::Value x = Ort::Value::CreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int64_t scale_shape = 1;

    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &num_frames, 1, &scale_shape, 1);

    int32_t language = 0;
    if (config_.model_config.sense_voice.language.empty()) {
      language = 0;
    } else if (meta_data.lang2id.count(
                   config_.model_config.sense_voice.language)) {
      language =
          meta_data.lang2id.at(config_.model_config.sense_voice.language);
    } else {
      SHERPA_ONNX_LOGE("Unknown language: %s. Use 0 instead.",
                       config_.model_config.sense_voice.language.c_str());
    }

    int32_t text_norm = config_.model_config.sense_voice.use_itn
                            ? meta_data.with_itn_id
                            : meta_data.without_itn_id;

    Ort::Value language_tensor =
        Ort::Value::CreateTensor(memory_info, &language, 1, &scale_shape, 1);

    Ort::Value text_norm_tensor =
        Ort::Value::CreateTensor(memory_info, &text_norm, 1, &scale_shape, 1);

    Ort::Value logits{nullptr};
    try {
      logits = model_->Forward(std::move(x), std::move(x_length),
                               std::move(language_tensor),
                               std::move(text_norm_tensor));
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE("\n\nCaught exception:\n\n%s\n\nReturn an empty result",
                       ex.what());
      return;
    }

    int64_t new_num_frames = num_frames + 4;
    Ort::Value logits_length = Ort::Value::CreateTensor(
        memory_info, &new_num_frames, 1, &scale_shape, 1);

    auto results =
        decoder_->Decode(std::move(logits), std::move(logits_length));

    /*
    for(int i=0; i< results.size(); i++){
      for(int j=0; j< results[i].tokens.size(); j++){
        SHERPA_ONNX_LOGE("token:%ld, timestamp:%d",
                       results[i].tokens[j],results[i].timestamps[j]);        
      }
    }
    */
    int32_t frame_shift_ms = 10;
    int32_t subsampling_factor = meta_data.window_shift;
    auto r = ConvertSenseVoiceResult(results[0], symbol_table_, frame_shift_ms,
                                     subsampling_factor);

    r.text = ApplyInverseTextNormalization(std::move(r.text));
    s->SetResult(r);
  }

  void InitFeatConfig() {
    const auto &meta_data = model_->GetModelMetadata();

    config_.feat_config.normalize_samples = meta_data.normalize_samples;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
    config_.feat_config.dither = 1.0f;
  }

  void LfrCmvn(std::vector<std::vector<float>> &vad_feats) const {
    const auto &meta_data = model_->GetModelMetadata();
    int32_t lfr_m = meta_data.window_size;
    int32_t lfr_n = meta_data.window_shift;
    int32_t in_feat_dim = config_.feat_config.feature_dim;

    std::vector<std::vector<float>> out_feats;
    int T = vad_feats.size();
    int T_lrf = ceil(1.0 * T / lfr_n);

    // Pad frames at start(copy first frame)
    for (int i = 0; i < (lfr_m - 1) / 2; i++) {
      vad_feats.insert(vad_feats.begin(), vad_feats[0]);
    }
    // Merge lfr_m frames as one,lfr_n frames per window
    T = T + (lfr_m - 1) / 2;
    std::vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
      if (lfr_m <= T - i * lfr_n) {
        for (int j = 0; j < lfr_m; j++) {
          p.insert(p.end(), vad_feats[i * lfr_n + j].begin(),
                   vad_feats[i * lfr_n + j].end());
        }
        out_feats.emplace_back(p);
        p.clear();
      } else {
        // Fill to lfr_m frames at last window if less than lfr_m frames  (copy
        // last frame)
        int num_padding = lfr_m - (T - i * lfr_n);
        for (int j = 0; j < (vad_feats.size() - i * lfr_n); j++) {
          p.insert(p.end(), vad_feats[i * lfr_n + j].begin(),
                   vad_feats[i * lfr_n + j].end());
        }
        for (int j = 0; j < num_padding; j++) {
          p.insert(p.end(), vad_feats[vad_feats.size() - 1].begin(),
                   vad_feats[vad_feats.size() - 1].end());
        }
        out_feats.emplace_back(p);
        p.clear();
      }
    }

    // Apply cmvn
    const std::vector<float> &neg_mean = meta_data.neg_mean;
    const std::vector<float> &inv_stddev = meta_data.inv_stddev;

    for (auto &out_feat : out_feats) {
      for (int j = 0; j < neg_mean.size(); j++) {
        out_feat[j] = (out_feat[j] + neg_mean[j]) * inv_stddev[j];
      }
    }
    vad_feats = out_feats;
  }

  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineSenseVoiceModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_SENSE_VOICE_IMPL_H_
