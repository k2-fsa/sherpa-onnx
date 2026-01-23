// sherpa-onnx/csrc/offline-recognizer-transducer-nemo-impl.h
//
// Copyright (c)  2022-2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_

#include <fstream>
#include <ios>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/offline-transducer-greedy-search-nemo-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-modified-beam-search-nemo-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-nemo-model.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transpose.h"
#include "sherpa-onnx/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

// defined in ./offline-recognizer-transducer-impl.h
OfflineRecognitionResult Convert(const OfflineTransducerDecoderResult &src,
                                 const SymbolTable &sym_table,
                                 int32_t frame_shift_ms,
                                 int32_t subsampling_factor);

class OfflineRecognizerTransducerNeMoImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerTransducerNeMoImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerNeMoModel>(
            config_.model_config)) {
    
    if (symbol_table_.Contains("<unk>")) {
      unk_id_ = symbol_table_["<unk>"];
    }
    
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty, model_->IsTDT());
    } else if (config_.decoding_method == "modified_beam_search") {
      // Initialize BPE encoder if provided
      if (!config_.model_config.bpe_vocab.empty()) {
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
            config_.model_config.bpe_vocab);
      }

      // Initialize hotwords if provided
      if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchNeMoDecoder>(
          model_.get(), config_.max_active_paths, unk_id_,
          config_.blank_penalty, model_->IsTDT(), config_.hotwords_score);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    PostInit();
  }

  template <typename Manager>
  explicit OfflineRecognizerTransducerNeMoImpl(
      Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineTransducerNeMoModel>(
            mgr, config_.model_config)) {
    
    if (symbol_table_.Contains("<unk>")) {
      unk_id_ = symbol_table_["<unk>"];
    }
    
    if (config_.decoding_method == "greedy_search") {
      decoder_ = std::make_unique<OfflineTransducerGreedySearchNeMoDecoder>(
          model_.get(), config_.blank_penalty, model_->IsTDT());
    } else if (config_.decoding_method == "modified_beam_search") {
      // Initialize BPE encoder if provided
      if (!config_.model_config.bpe_vocab.empty()) {
        auto buf = ReadFile(mgr, config_.model_config.bpe_vocab);
        std::istringstream iss(std::string(buf.begin(), buf.end()));
        bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(iss);
      }

      // Initialize hotwords if provided
      if (!config_.hotwords_file.empty()) {
        InitHotwords(mgr);
      }

      decoder_ = std::make_unique<OfflineTransducerModifiedBeamSearchNeMoDecoder>(
          model_.get(), config_.max_active_paths, unk_id_,
          config_.blank_penalty, model_->IsTDT(), config_.hotwords_score);
    } else {
      SHERPA_ONNX_LOGE("Unsupported decoding method: %s",
                       config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    PostInit();
  }

  std::unique_ptr<OfflineStream> CreateStream(
      const std::string &hotwords) const override {
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    std::vector<float> current_scores;
    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &current, &current_scores)) {
      SHERPA_ONNX_LOGE("Encode hotwords failed, skipping, hotwords are : '%s'",
                       hotwords.c_str());
    }

    int32_t num_default_hws = hotwords_.size();
    int32_t num_hws = current.size();

    current.insert(current.end(), hotwords_.begin(), hotwords_.end());

    if (!current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else if (!current_scores.empty() && boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_default_hws,
                            config_.hotwords_score);
    } else if (current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_hws,
                            config_.hotwords_score);
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else {
      // Do nothing.
    }

    auto context_graph = std::make_shared<ContextGraph>(
        current, config_.hotwords_score, current_scores);
    return std::make_unique<OfflineStream>(config_.feat_config, context_graph);
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config,
                                           hotwords_graph_);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<Ort::Value> features;

    features.reserve(n);

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int64_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      auto f = ss[i]->GetFrames();
      int32_t num_frames = f.size() / feat_dim;

      features_length_vec[i] = num_frames;
      features_vec[i] = std::move(f);

      std::array<int64_t, 2> shape = {num_frames, feat_dim};

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

    Ort::Value x = PadSequence(model_->Allocator(), features_pointer, 0);

    auto t = model_->RunEncoder(std::move(x), std::move(x_length));
    // t[0] encoder_out, float tensor, (batch_size, dim, T)
    // t[1] encoder_out_length, int64 tensor, (batch_size,)

    Ort::Value encoder_out = Transpose12(model_->Allocator(), &t[0]);

    auto results = decoder_->Decode(std::move(encoder_out), std::move(t[1]),
                                    ss, n);

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());

      // Remove leading space from BPE tokenization
      if (!r.text.empty() && r.text.front() == ' ') {
        r.text.erase(0, 1);
      }

      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));

      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void PostInit() {
    int32_t feat_dim = model_->FeatureDim();

    if (feat_dim > 0) {
      config_.feat_config.feature_dim = feat_dim;
    }

    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    if (model_->IsGigaAM()) {
      config_.feat_config.low_freq = 0;
      config_.feat_config.high_freq = 8000;
      config_.feat_config.remove_dc_offset = false;
      config_.feat_config.preemph_coeff = 0;
      config_.feat_config.window_type = "hann";
      config_.feat_config.feature_dim = 64;

      // see
      // https://github.com/salute-developers/GigaAM/blob/main/gigaam/preprocess.py#L68
      //
      // GigaAM uses n_fft 400
      config_.feat_config.round_to_power_of_two = false;
    } else {
      config_.feat_config.low_freq = 0;
      // config_.feat_config.high_freq = 8000;
      config_.feat_config.is_librosa = true;
      config_.feat_config.remove_dc_offset = false;
      // config_.feat_config.window_type = "hann";
    }

    int32_t vocab_size = model_->VocabSize();

    // check the blank ID
    if (!symbol_table_.Contains("<blk>")) {
      SHERPA_ONNX_LOGE("tokens.txt does not include the blank token <blk>");
      SHERPA_ONNX_EXIT(-1);
    }

    if (symbol_table_["<blk>"] != vocab_size - 1) {
      SHERPA_ONNX_LOGE("<blk> is not the last token!");
      SHERPA_ONNX_EXIT(-1);
    }

    if (symbol_table_.NumSymbols() != vocab_size) {
      SHERPA_ONNX_LOGE("number of lines in tokens.txt %d != %d (vocab_size)",
                       symbol_table_.NumSymbols(), vocab_size);
      SHERPA_ONNX_EXIT(-1);
    }
  }

  void InitHotwords() {
    // each line in hotwords_file contains space-separated words

    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: '%s'",
                       config_.hotwords_file.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Some hotwords failed to encode and were skipped. See above for "
          "details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  template <typename Manager>
  void InitHotwords(Manager *mgr) {
    // each line in hotwords_file contains space-separated words

    auto buf = ReadFile(mgr, config_.hotwords_file);

    std::istringstream is(std::string(buf.begin(), buf.end()));

    if (!EncodeHotwords(is, config_.model_config.modeling_unit, symbol_table_,
                        bpe_encoder_.get(), &hotwords_, &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Some hotwords failed to encode and were skipped. See above for "
          "details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;
  std::unique_ptr<OfflineTransducerNeMoModel> model_;
  std::unique_ptr<OfflineTransducerDecoder> decoder_;
  int32_t unk_id_ = -1;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_TRANSDUCER_NEMO_IMPL_H_
