// wasm/combined/sherpa-onnx-wasm-combined.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <stdio.h>
#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

// This is a combined implementation that provides all the necessary C functions
// for the WASM module, incorporating debug printing for all supported features.

extern "C" {

// ============================================================================
// Verify memory layouts with static assertions
// ============================================================================

// ASR memory layout verification
static_assert(sizeof(SherpaOnnxOnlineTransducerModelConfig) == 3 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineParaformerModelConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineZipformer2CtcModelConfig) == 1 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineModelConfig) ==
                  sizeof(SherpaOnnxOnlineTransducerModelConfig) +
                      sizeof(SherpaOnnxOnlineParaformerModelConfig) +
                      sizeof(SherpaOnnxOnlineZipformer2CtcModelConfig) + 9 * 4,
              "");
static_assert(sizeof(SherpaOnnxFeatureConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineCtcFstDecoderConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOnlineRecognizerConfig) ==
                  sizeof(SherpaOnnxFeatureConfig) +
                      sizeof(SherpaOnnxOnlineModelConfig) + 8 * 4 +
                      sizeof(SherpaOnnxOnlineCtcFstDecoderConfig) + 5 * 4,
              "");

// VAD memory layout verification
static_assert(sizeof(SherpaOnnxSileroVadModelConfig) == 6 * 4, "");
static_assert(sizeof(SherpaOnnxVadModelConfig) ==
                  sizeof(SherpaOnnxSileroVadModelConfig) + 4 * 4,
              "");

// TTS memory layout verification
static_assert(sizeof(SherpaOnnxOfflineTtsVitsModelConfig) == 8 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsMatchaModelConfig) == 8 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsKokoroModelConfig) == 7 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTtsModelConfig) ==
                  sizeof(SherpaOnnxOfflineTtsVitsModelConfig) +
                      sizeof(SherpaOnnxOfflineTtsMatchaModelConfig) +
                      sizeof(SherpaOnnxOfflineTtsKokoroModelConfig) + 3 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineTtsConfig) ==
                  sizeof(SherpaOnnxOfflineTtsModelConfig) + 4 * 4,
              "");

// Speaker Diarization memory layout verification
static_assert(sizeof(SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig) ==
                  1 * 4,
              "");
static_assert(
    sizeof(SherpaOnnxOfflineSpeakerSegmentationModelConfig) ==
        sizeof(SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig) + 3 * 4,
    "");
static_assert(sizeof(SherpaOnnxFastClusteringConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxSpeakerEmbeddingExtractorConfig) == 4 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineSpeakerDiarizationConfig) ==
                  sizeof(SherpaOnnxOfflineSpeakerSegmentationModelConfig) +
                      sizeof(SherpaOnnxSpeakerEmbeddingExtractorConfig) +
                      sizeof(SherpaOnnxFastClusteringConfig) + 2 * 4,
              "");

// Speech Enhancement memory layout verification
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig) == 1 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserModelConfig) ==
                  sizeof(SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig) +
                      3 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineSpeechDenoiserConfig) ==
                  sizeof(SherpaOnnxOfflineSpeechDenoiserModelConfig),
              "");

// Keyword Spotting memory layout verification
static_assert(sizeof(SherpaOnnxKeywordSpotterConfig) ==
                  sizeof(SherpaOnnxFeatureConfig) +
                      sizeof(SherpaOnnxOnlineModelConfig) + 7 * 4,
              "");

// ============================================================================
// Debug printing functions for all model types
// ============================================================================

// Helper function to copy between heap locations
void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}

// Debug printing for Online ASR configuration
void MyPrintOnlineASR(SherpaOnnxOnlineRecognizerConfig *config) {
  auto model_config = &config->model_config;
  auto feat = &config->feat_config;
  auto transducer_model_config = &model_config->transducer;
  auto paraformer_model_config = &model_config->paraformer;
  auto ctc_model_config = &model_config->zipformer2_ctc;

  fprintf(stdout, "----------Online ASR Configuration----------\n");
  fprintf(stdout, "----------online transducer model config----------\n");
  fprintf(stdout, "encoder: %s\n", transducer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", transducer_model_config->decoder);
  fprintf(stdout, "joiner: %s\n", transducer_model_config->joiner);

  fprintf(stdout, "----------online parformer model config----------\n");
  fprintf(stdout, "encoder: %s\n", paraformer_model_config->encoder);
  fprintf(stdout, "decoder: %s\n", paraformer_model_config->decoder);

  fprintf(stdout, "----------online ctc model config----------\n");
  fprintf(stdout, "model: %s\n", ctc_model_config->model);
  fprintf(stdout, "tokens: %s\n", model_config->tokens);
  fprintf(stdout, "num_threads: %d\n", model_config->num_threads);
  fprintf(stdout, "provider: %s\n", model_config->provider);
  fprintf(stdout, "debug: %d\n", model_config->debug);
  fprintf(stdout, "model type: %s\n", model_config->model_type);
  fprintf(stdout, "modeling unit: %s\n", model_config->modeling_unit);
  fprintf(stdout, "bpe vocab: %s\n", model_config->bpe_vocab);
  fprintf(stdout, "tokens_buf: %s\n",
          model_config->tokens_buf ? model_config->tokens_buf : "");
  fprintf(stdout, "tokens_buf_size: %d\n", model_config->tokens_buf_size);

  fprintf(stdout, "----------feat config----------\n");
  fprintf(stdout, "sample rate: %d\n", feat->sample_rate);
  fprintf(stdout, "feat dim: %d\n", feat->feature_dim);

  fprintf(stdout, "----------recognizer config----------\n");
  fprintf(stdout, "decoding method: %s\n", config->decoding_method);
  fprintf(stdout, "max active paths: %d\n", config->max_active_paths);
  fprintf(stdout, "enable_endpoint: %d\n", config->enable_endpoint);
  fprintf(stdout, "rule1_min_trailing_silence: %.2f\n",
          config->rule1_min_trailing_silence);
  fprintf(stdout, "rule2_min_trailing_silence: %.2f\n",
          config->rule2_min_trailing_silence);
  fprintf(stdout, "rule3_min_utterance_length: %.2f\n",
          config->rule3_min_utterance_length);
  fprintf(stdout, "hotwords_file: %s\n", config->hotwords_file);
  fprintf(stdout, "hotwords_score: %.2f\n", config->hotwords_score);
  fprintf(stdout, "rule_fsts: %s\n", config->rule_fsts);
  fprintf(stdout, "rule_fars: %s\n", config->rule_fars);
  fprintf(stdout, "blank_penalty: %f\n", config->blank_penalty);

  fprintf(stdout, "----------ctc fst decoder config----------\n");
  fprintf(stdout, "graph: %s\n", config->ctc_fst_decoder_config.graph);
  fprintf(stdout, "max_active: %d\n",
          config->ctc_fst_decoder_config.max_active);
}

// Debug printing for VAD configuration
void MyPrintVAD(SherpaOnnxVadModelConfig *config) {
  auto silero_vad = &config->silero_vad;

  fprintf(stdout, "----------Voice Activity Detection Configuration----------\n");
  fprintf(stdout, "----------silero_vad config----------\n");
  fprintf(stdout, "model: %s\n", silero_vad->model);
  fprintf(stdout, "threshold: %.3f\n", silero_vad->threshold);
  fprintf(stdout, "min_silence_duration: %.3f\n",
          silero_vad->min_silence_duration);
  fprintf(stdout, "min_speech_duration: %.3f\n",
          silero_vad->min_speech_duration);
  fprintf(stdout, "window_size: %d\n", silero_vad->window_size);
  fprintf(stdout, "max_speech_duration: %.3f\n",
          silero_vad->max_speech_duration);

  fprintf(stdout, "----------config----------\n");
  fprintf(stdout, "sample_rate: %d\n", config->sample_rate);
  fprintf(stdout, "num_threads: %d\n", config->num_threads);
  fprintf(stdout, "provider: %s\n", config->provider);
  fprintf(stdout, "debug: %d\n", config->debug);
}

// Debug printing for TTS configuration
void MyPrintTTS(SherpaOnnxOfflineTtsConfig *tts_config) {
  auto tts_model_config = &tts_config->model;
  auto vits_model_config = &tts_model_config->vits;
  auto matcha_model_config = &tts_model_config->matcha;
  auto kokoro = &tts_model_config->kokoro;

  fprintf(stdout, "----------Text-to-Speech Configuration----------\n");
  fprintf(stdout, "----------vits model config----------\n");
  fprintf(stdout, "model: %s\n", vits_model_config->model);
  fprintf(stdout, "lexicon: %s\n", vits_model_config->lexicon);
  fprintf(stdout, "tokens: %s\n", vits_model_config->tokens);
  fprintf(stdout, "data_dir: %s\n", vits_model_config->data_dir);
  fprintf(stdout, "noise scale: %.3f\n", vits_model_config->noise_scale);
  fprintf(stdout, "noise scale w: %.3f\n", vits_model_config->noise_scale_w);
  fprintf(stdout, "length scale: %.3f\n", vits_model_config->length_scale);
  fprintf(stdout, "dict_dir: %s\n", vits_model_config->dict_dir);

  fprintf(stdout, "----------matcha model config----------\n");
  fprintf(stdout, "acoustic_model: %s\n", matcha_model_config->acoustic_model);
  fprintf(stdout, "vocoder: %s\n", matcha_model_config->vocoder);
  fprintf(stdout, "lexicon: %s\n", matcha_model_config->lexicon);
  fprintf(stdout, "tokens: %s\n", matcha_model_config->tokens);
  fprintf(stdout, "data_dir: %s\n", matcha_model_config->data_dir);
  fprintf(stdout, "noise scale: %.3f\n", matcha_model_config->noise_scale);
  fprintf(stdout, "length scale: %.3f\n", matcha_model_config->length_scale);
  fprintf(stdout, "dict_dir: %s\n", matcha_model_config->dict_dir);

  fprintf(stdout, "----------kokoro model config----------\n");
  fprintf(stdout, "model: %s\n", kokoro->model);
  fprintf(stdout, "voices: %s\n", kokoro->voices);
  fprintf(stdout, "tokens: %s\n", kokoro->tokens);
  fprintf(stdout, "data_dir: %s\n", kokoro->data_dir);
  fprintf(stdout, "length scale: %.3f\n", kokoro->length_scale);
  fprintf(stdout, "dict_dir: %s\n", kokoro->dict_dir);
  fprintf(stdout, "lexicon: %s\n", kokoro->lexicon);

  fprintf(stdout, "----------tts model config----------\n");
  fprintf(stdout, "num threads: %d\n", tts_model_config->num_threads);
  fprintf(stdout, "debug: %d\n", tts_model_config->debug);
  fprintf(stdout, "provider: %s\n", tts_model_config->provider);

  fprintf(stdout, "----------tts config----------\n");
  fprintf(stdout, "rule_fsts: %s\n", tts_config->rule_fsts);
  fprintf(stdout, "rule_fars: %s\n", tts_config->rule_fars);
  fprintf(stdout, "max num sentences: %d\n", tts_config->max_num_sentences);
  fprintf(stdout, "silence scale: %.3f\n", tts_config->silence_scale);
}

// Debug printing for Speaker Diarization configuration
void MyPrintSpeakerDiarization(const SherpaOnnxOfflineSpeakerDiarizationConfig *sd_config) {
  const auto &segmentation = sd_config->segmentation;
  const auto &embedding = sd_config->embedding;
  const auto &clustering = sd_config->clustering;

  fprintf(stdout, "----------Speaker Diarization Configuration----------\n");
  fprintf(stdout, "----------segmentation config----------\n");
  fprintf(stdout, "pyannote model: %s\n", segmentation.pyannote.model);
  fprintf(stdout, "num threads: %d\n", segmentation.num_threads);
  fprintf(stdout, "debug: %d\n", segmentation.debug);
  fprintf(stdout, "provider: %s\n", segmentation.provider);

  fprintf(stdout, "----------embedding config----------\n");
  fprintf(stdout, "model: %s\n", embedding.model);
  fprintf(stdout, "num threads: %d\n", embedding.num_threads);
  fprintf(stdout, "debug: %d\n", embedding.debug);
  fprintf(stdout, "provider: %s\n", embedding.provider);

  fprintf(stdout, "----------clustering config----------\n");
  fprintf(stdout, "num_clusters: %d\n", clustering.num_clusters);
  fprintf(stdout, "threshold: %.3f\n", clustering.threshold);

  fprintf(stdout, "min_duration_on: %.3f\n", sd_config->min_duration_on);
  fprintf(stdout, "min_duration_off: %.3f\n", sd_config->min_duration_off);
}

// Debug printing for Speech Enhancement configuration
void MyPrintSpeechEnhancement(SherpaOnnxOfflineSpeechDenoiserConfig *config) {
  auto model = &config->model;
  auto gtcrn = &model->gtcrn;

  fprintf(stdout, "----------Speech Enhancement Configuration----------\n");
  fprintf(stdout, "----------offline speech denoiser model config----------\n");
  fprintf(stdout, "gtcrn: %s\n", gtcrn->model);
  fprintf(stdout, "num threads: %d\n", model->num_threads);
  fprintf(stdout, "debug: %d\n", model->debug);
  fprintf(stdout, "provider: %s\n", model->provider);
}

// Debug printing for Keyword Spotting configuration
void MyPrintKeywordSpotting(SherpaOnnxKeywordSpotterConfig *config) {
  auto feat = &config->feat_config;
  auto model = &config->model_config;
  auto transducer = &model->transducer;

  fprintf(stdout, "----------Keyword Spotting Configuration----------\n");
  fprintf(stdout, "model_config.transducer.encoder: %s\n", transducer->encoder);
  fprintf(stdout, "model_config.transducer.decoder: %s\n", transducer->decoder);
  fprintf(stdout, "model_config.transducer.joiner: %s\n", transducer->joiner);
  fprintf(stdout, "model_config.tokens: %s\n", model->tokens);
  fprintf(stdout, "model_config.num_threads: %d\n", model->num_threads);
  fprintf(stdout, "model_config.provider: %s\n", model->provider);
  fprintf(stdout, "model_config.debug: %d\n", model->debug);
  
  fprintf(stdout, "feat_config.sample_rate: %d\n", feat->sample_rate);
  fprintf(stdout, "feat_config.feature_dim: %d\n", feat->feature_dim);
  
  fprintf(stdout, "max_active_paths: %d\n", config->max_active_paths);
  fprintf(stdout, "num_trailing_blanks: %d\n", config->num_trailing_blanks);
  fprintf(stdout, "keywords_score: %.3f\n", config->keywords_score);
  fprintf(stdout, "keywords_threshold: %.3f\n", config->keywords_threshold);
  fprintf(stdout, "keywords_file: %s\n", config->keywords_file ? config->keywords_file : "");
}

} // extern "C" 