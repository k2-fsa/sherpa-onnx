// sherpa-onnx/csrc/offline-tts-vits-impl.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include <strstream>

#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif
#include "kaldifst/csrc/text-normalizer.h"
#include "sherpa-onnx/csrc/lexicon.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-character-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflineTtsVitsImpl : public OfflineTtsImpl {
 public:
  explicit OfflineTtsVitsImpl(const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(config.model)) {
    InitFrontend();

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
        }
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(f));
      }
    }
  }

#if __ANDROID_API__ >= 9
  OfflineTtsVitsImpl(AAssetManager *mgr, const OfflineTtsConfig &config)
      : config_(config),
        model_(std::make_unique<OfflineTtsVitsModel>(mgr, config.model)) {
    InitFrontend(mgr);

    if (!config.rule_fsts.empty()) {
      std::vector<std::string> files;
      SplitStringToVector(config.rule_fsts, ",", false, &files);
      tn_list_.reserve(files.size());
      for (const auto &f : files) {
        if (config.model.debug) {
          SHERPA_ONNX_LOGE("rule fst: %s", f.c_str());
        }
        auto buf = ReadFile(mgr, f);
        std::istrstream is(buf.data(), buf.size());
        tn_list_.push_back(std::make_unique<kaldifst::TextNormalizer>(is));
      }
    }
  }
#endif

  int32_t SampleRate() const override {
    return model_->GetMetaData().sample_rate;
  }

  int32_t NumSpeakers() const override {
    return model_->GetMetaData().num_speakers;
  }

  GeneratedAudio Generate(
      const std::string &_text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    const auto &meta_data = model_->GetMetaData();
    int32_t num_speakers = meta_data.num_speakers;

    if (num_speakers == 0 && sid != 0) {
      SHERPA_ONNX_LOGE(
          "This is a single-speaker model and supports only sid 0. Given sid: "
          "%d. sid is ignored",
          static_cast<int32_t>(sid));
    }

    if (num_speakers != 0 && (sid >= num_speakers || sid < 0)) {
      SHERPA_ONNX_LOGE(
          "This model contains only %d speakers. sid should be in the range "
          "[%d, %d]. Given: %d. Use sid=0",
          num_speakers, 0, num_speakers - 1, static_cast<int32_t>(sid));
      sid = 0;
    }

    std::string text = _text;
    if (config_.model.debug) {
      SHERPA_ONNX_LOGE("Raw text: %s", text.c_str());
    }

    if (!tn_list_.empty()) {
      for (const auto &tn : tn_list_) {
        text = tn->Normalize(text);
        if (config_.model.debug) {
          SHERPA_ONNX_LOGE("After normalizing: %s", text.c_str());
        }
      }
    }

    std::vector<std::vector<int64_t>> x =
        frontend_->ConvertTextToTokenIds(text, meta_data.voice);

    if (x.empty() || (x.size() == 1 && x[0].empty())) {
      SHERPA_ONNX_LOGE("Failed to convert %s to token IDs", text.c_str());
      return {};
    }

    // TODO(fangjun): add blank inside the frontend, not here
    if (meta_data.add_blank && config_.model.vits.data_dir.empty() &&
        meta_data.frontend != "characters") {
      for (auto &k : x) {
        k = AddBlank(k);
      }
    }

    int32_t x_size = static_cast<int32_t>(x.size());

    if (config_.max_num_sentences <= 0 || x_size <= config_.max_num_sentences) {
      auto ans = Process(x, sid, speed);
      if (callback) {
        callback(ans.samples.data(), ans.samples.size());
      }
      return ans;
    }

    // the input text is too long, we process sentences within it in batches
    // to avoid OOM. Batch size is config_.max_num_sentences
    std::vector<std::vector<int64_t>> batch;
    int32_t batch_size = config_.max_num_sentences;
    batch.reserve(config_.max_num_sentences);
    int32_t num_batches = x_size / batch_size;

    if (config_.model.debug) {
      SHERPA_ONNX_LOGE(
          "Text is too long. Split it into %d batches. batch size: %d. Number "
          "of sentences: %d",
          num_batches, batch_size, x_size);
    }

    GeneratedAudio ans;

    int32_t k = 0;

    for (int32_t b = 0; b != num_batches; ++b) {
      batch.clear();
      for (int32_t i = 0; i != batch_size; ++i, ++k) {
        batch.push_back(std::move(x[k]));
      }

      auto audio = Process(batch, sid, speed);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        callback(audio.samples.data(), audio.samples.size());
        // Caution(fangjun): audio is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }
    }

    batch.clear();
    while (k < x.size()) {
      batch.push_back(std::move(x[k]));
      ++k;
    }

    if (!batch.empty()) {
      auto audio = Process(batch, sid, speed);
      ans.sample_rate = audio.sample_rate;
      ans.samples.insert(ans.samples.end(), audio.samples.begin(),
                         audio.samples.end());
      if (callback) {
        callback(audio.samples.data(), audio.samples.size());
        // Caution(fangjun): audio is freed when the callback returns, so users
        // should copy the data if they want to access the data after
        // the callback returns to avoid segmentation fault.
      }
    }

    return ans;
  }

 private:
#if __ANDROID_API__ >= 9
  void InitFrontend(AAssetManager *mgr) {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.frontend == "characters") {
      frontend_ = std::make_unique<OfflineTtsCharacterFrontend>(
          mgr, config_.model.vits.tokens, meta_data);
    } else if ((meta_data.is_piper || meta_data.is_coqui) &&
               !config_.model.vits.data_dir.empty()) {
      frontend_ = std::make_unique<PiperPhonemizeLexicon>(
          mgr, config_.model.vits.tokens, config_.model.vits.data_dir,
          meta_data);
    } else {
      if (config_.model.vits.lexicon.empty()) {
        SHERPA_ONNX_LOGE(
            "Not a model using characters as modeling unit. Please provide "
            "--vits-lexicon if you leave --vits-data-dir empty");
        exit(-1);
      }

      frontend_ = std::make_unique<Lexicon>(
          mgr, config_.model.vits.lexicon, config_.model.vits.tokens,
          meta_data.punctuations, meta_data.language, config_.model.debug);
    }
  }
#endif

  void InitFrontend() {
    const auto &meta_data = model_->GetMetaData();

    if (meta_data.frontend == "characters") {
      frontend_ = std::make_unique<OfflineTtsCharacterFrontend>(
          config_.model.vits.tokens, meta_data);
    } else if ((meta_data.is_piper || meta_data.is_coqui) &&
               !config_.model.vits.data_dir.empty()) {
      frontend_ = std::make_unique<PiperPhonemizeLexicon>(
          config_.model.vits.tokens, config_.model.vits.data_dir,
          model_->GetMetaData());
    } else {
      if (config_.model.vits.lexicon.empty()) {
        SHERPA_ONNX_LOGE(
            "Not a model using characters as modeling unit. Please provide "
            "--vits-lexicon if you leave --vits-data-dir empty");
        exit(-1);
      }
      frontend_ = std::make_unique<Lexicon>(
          config_.model.vits.lexicon, config_.model.vits.tokens,
          meta_data.punctuations, meta_data.language, config_.model.debug);
    }
  }

  std::vector<int64_t> AddBlank(const std::vector<int64_t> &x) const {
    // we assume the blank ID is 0
    std::vector<int64_t> buffer(x.size() * 2 + 1);
    int32_t i = 1;
    for (auto k : x) {
      buffer[i] = k;
      i += 2;
    }
    return buffer;
  }

  GeneratedAudio Process(const std::vector<std::vector<int64_t>> &tokens,
                         int32_t sid, float speed) const {
    int32_t num_tokens = 0;
    for (const auto &k : tokens) {
      num_tokens += k.size();
    }

    std::vector<int64_t> x;
    x.reserve(num_tokens);
    for (const auto &k : tokens) {
      x.insert(x.end(), k.begin(), k.end());
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 2> x_shape = {1, static_cast<int32_t>(x.size())};
    Ort::Value x_tensor = Ort::Value::CreateTensor(
        memory_info, x.data(), x.size(), x_shape.data(), x_shape.size());

    Ort::Value audio = model_->Run(std::move(x_tensor), sid, speed);

    std::vector<int64_t> audio_shape =
        audio.GetTensorTypeAndShapeInfo().GetShape();

    int64_t total = 1;
    // The output shape may be (1, 1, total) or (1, total) or (total,)
    for (auto i : audio_shape) {
      total *= i;
    }

    const float *p = audio.GetTensorData<float>();

    GeneratedAudio ans;
    ans.sample_rate = model_->GetMetaData().sample_rate;
    ans.samples = std::vector<float>(p, p + total);
    return ans;
  }

 private:
  OfflineTtsConfig config_;
  std::unique_ptr<OfflineTtsVitsModel> model_;
  std::vector<std::unique_ptr<kaldifst::TextNormalizer>> tn_list_;
  std::unique_ptr<OfflineTtsFrontend> frontend_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_VITS_IMPL_H_
