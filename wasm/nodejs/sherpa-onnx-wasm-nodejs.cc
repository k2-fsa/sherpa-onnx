// wasm/sherpa-onnx-wasm-main-nodejs.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include <stdio.h>

#include <algorithm>
#include <memory>

#include "sherpa-onnx/c-api/c-api.h"

extern "C" {

static_assert(sizeof(SherpaOnnxOfflineTransducerModelConfig) == 3 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineParaformerModelConfig) == 4, "");

static_assert(sizeof(SherpaOnnxOfflineZipformerCtcModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineWenetCtcModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineOmnilingualAsrCtcModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineMedAsrCtcModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineFireRedAsrCtcModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineFunASRNanoModelConfig) == 13 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineQwen3ASRModelConfig) == 10 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineCohereTranscribeModelConfig) == 5 * 4,
              "");
static_assert(sizeof(SherpaOnnxOfflineDolphinModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineNemoEncDecCtcModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineWhisperModelConfig) == 7 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineFireRedAsrModelConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineMoonshineModelConfig) == 5 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineTdnnModelConfig) == 4, "");
static_assert(sizeof(SherpaOnnxOfflineSenseVoiceModelConfig) == 3 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineCanaryModelConfig) == 5 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineLMConfig) == 2 * 4, "");

static_assert(sizeof(SherpaOnnxOfflineModelConfig) ==
                  sizeof(SherpaOnnxOfflineTransducerModelConfig) +
                      sizeof(SherpaOnnxOfflineParaformerModelConfig) +
                      sizeof(SherpaOnnxOfflineNemoEncDecCtcModelConfig) +
                      sizeof(SherpaOnnxOfflineWhisperModelConfig) +
                      sizeof(SherpaOnnxOfflineTdnnModelConfig) + 8 * 4 +
                      sizeof(SherpaOnnxOfflineSenseVoiceModelConfig) +
                      sizeof(SherpaOnnxOfflineMoonshineModelConfig) +
                      sizeof(SherpaOnnxOfflineFireRedAsrModelConfig) +
                      sizeof(SherpaOnnxOfflineDolphinModelConfig) +
                      sizeof(SherpaOnnxOfflineZipformerCtcModelConfig) +
                      sizeof(SherpaOnnxOfflineCanaryModelConfig) +
                      sizeof(SherpaOnnxOfflineWenetCtcModelConfig) +
                      sizeof(SherpaOnnxOfflineOmnilingualAsrCtcModelConfig) +
                      sizeof(SherpaOnnxOfflineMedAsrCtcModelConfig) +
                      sizeof(SherpaOnnxOfflineFunASRNanoModelConfig) +
                      sizeof(SherpaOnnxOfflineFireRedAsrCtcModelConfig) +
                      sizeof(SherpaOnnxOfflineQwen3ASRModelConfig) +
                      sizeof(SherpaOnnxOfflineCohereTranscribeModelConfig),

              "");
static_assert(sizeof(SherpaOnnxFeatureConfig) == 2 * 4, "");
static_assert(sizeof(SherpaOnnxOfflineRecognizerConfig) ==
                  sizeof(SherpaOnnxFeatureConfig) +
                      sizeof(SherpaOnnxOfflineLMConfig) +
                      sizeof(SherpaOnnxOfflineModelConfig) + 7 * 4 +
                      sizeof(SherpaOnnxHomophoneReplacerConfig),
              "");

void PrintOfflineTtsConfig(SherpaOnnxOfflineTtsConfig *tts_config) {
  auto tts_model_config = &tts_config->model;
  auto vits_model_config = &tts_model_config->vits;
  auto matcha_model_config = &tts_model_config->matcha;
  auto kokoro = &tts_model_config->kokoro;
  auto kitten = &tts_model_config->kitten;
  auto zipvoice = &tts_model_config->zipvoice;
  auto pocket = &tts_model_config->pocket;
  auto supertonic = &tts_model_config->supertonic;

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
  fprintf(stdout, "lang: %s\n", kokoro->lang);

  fprintf(stdout, "----------kitten model config----------\n");
  fprintf(stdout, "model: %s\n", kitten->model);
  fprintf(stdout, "voices: %s\n", kitten->voices);
  fprintf(stdout, "tokens: %s\n", kitten->tokens);
  fprintf(stdout, "data_dir: %s\n", kitten->data_dir);
  fprintf(stdout, "length scale: %.3f\n", kitten->length_scale);

  fprintf(stdout, "----------zipvoice model config----------\n");
  fprintf(stdout, "tokens: %s\n", zipvoice->tokens);
  fprintf(stdout, "encoder: %s\n", zipvoice->encoder);
  fprintf(stdout, "decoder: %s\n", zipvoice->decoder);
  fprintf(stdout, "vocoder: %s\n", zipvoice->vocoder);
  fprintf(stdout, "data_dir: %s\n", zipvoice->data_dir);
  fprintf(stdout, "lexicon: %s\n", zipvoice->lexicon);
  fprintf(stdout, "feat scale: %.3f\n", zipvoice->feat_scale);
  fprintf(stdout, "t_shift: %.3f\n", zipvoice->t_shift);
  fprintf(stdout, "target_rms: %.3f\n", zipvoice->target_rms);
  fprintf(stdout, "guidance_scale: %.3f\n", zipvoice->guidance_scale);

  fprintf(stdout, "----------pocketTTS model config----------\n");
  fprintf(stdout, "lm_flow: %s\n", pocket->lm_flow);
  fprintf(stdout, "lm_main: %s\n", pocket->lm_main);
  fprintf(stdout, "encoder: %s\n", pocket->encoder);
  fprintf(stdout, "decoder: %s\n", pocket->decoder);
  fprintf(stdout, "text_conditioner: %s\n", pocket->text_conditioner);
  fprintf(stdout, "vocab_json: %s\n", pocket->vocab_json);
  fprintf(stdout, "token_scores_json: %s\n", pocket->token_scores_json);
  fprintf(stdout, "voice_embedding_cache_capacity: %d\n",
          pocket->voice_embedding_cache_capacity);

  fprintf(stdout, "----------supertonic model config----------\n");
  fprintf(stdout, "duration_predictor: %s\n", supertonic->duration_predictor);
  fprintf(stdout, "text_encoder: %s\n", supertonic->text_encoder);
  fprintf(stdout, "vector_estimator: %s\n", supertonic->vector_estimator);
  fprintf(stdout, "vocoder: %s\n", supertonic->vocoder);
  fprintf(stdout, "tts_json: %s\n", supertonic->tts_json);
  fprintf(stdout, "unicode_indexer: %s\n", supertonic->unicode_indexer);
  fprintf(stdout, "voice_style: %s\n", supertonic->voice_style);

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

void PrintOfflineRecognizerConfig(SherpaOnnxOfflineRecognizerConfig *config) {
  auto model_config = &config->model_config;
  auto feat = &config->feat_config;
  auto transducer = &model_config->transducer;
  auto paraformer = &model_config->paraformer;
  auto nemo_ctc = &model_config->nemo_ctc;
  auto whisper = &model_config->whisper;
  auto tdnn = &model_config->tdnn;
  auto sense_voice = &model_config->sense_voice;
  auto moonshine = &model_config->moonshine;
  auto fire_red_asr = &model_config->fire_red_asr;
  auto dolphin = &model_config->dolphin;
  auto zipformer_ctc = &model_config->zipformer_ctc;
  auto canary = &model_config->canary;
  auto wenet_ctc = &model_config->wenet_ctc;
  auto omnilingual = &model_config->omnilingual;
  auto medasr = &model_config->medasr;
  auto funasr_nano = &model_config->funasr_nano;
  auto fire_red_asr_ctc = &model_config->fire_red_asr_ctc;
  auto qwen3_asr = &model_config->qwen3_asr;
  auto cohere_transcribe = &model_config->cohere_transcribe;

  fprintf(stdout, "----------offline transducer model config----------\n");
  fprintf(stdout, "encoder: %s\n", transducer->encoder);
  fprintf(stdout, "decoder: %s\n", transducer->decoder);
  fprintf(stdout, "joiner: %s\n", transducer->joiner);

  fprintf(stdout, "----------offline paraformer model config----------\n");
  fprintf(stdout, "model: %s\n", paraformer->model);

  fprintf(stdout, "----------offline nemo_ctc model config----------\n");
  fprintf(stdout, "model: %s\n", nemo_ctc->model);

  fprintf(stdout, "----------offline whisper model config----------\n");
  fprintf(stdout, "encoder: %s\n", whisper->encoder);
  fprintf(stdout, "decoder: %s\n", whisper->decoder);
  fprintf(stdout, "language: %s\n", whisper->language);
  fprintf(stdout, "task: %s\n", whisper->task);
  fprintf(stdout, "tail_paddings: %d\n", whisper->tail_paddings);

  fprintf(stdout, "----------offline tdnn model config----------\n");
  fprintf(stdout, "model: %s\n", tdnn->model);

  fprintf(stdout, "----------offline sense_voice model config----------\n");
  fprintf(stdout, "model: %s\n", sense_voice->model);
  fprintf(stdout, "language: %s\n", sense_voice->language);
  fprintf(stdout, "use_itn: %d\n", sense_voice->use_itn);

  fprintf(stdout, "----------offline moonshine model config----------\n");
  fprintf(stdout, "preprocessor: %s\n", moonshine->preprocessor);
  fprintf(stdout, "encoder: %s\n", moonshine->encoder);
  fprintf(stdout, "uncached_decoder: %s\n", moonshine->uncached_decoder);
  fprintf(stdout, "cached_decoder: %s\n", moonshine->cached_decoder);
  fprintf(stdout, "merged_decoder: %s\n", moonshine->merged_decoder);

  fprintf(stdout, "----------offline FireRedAsr model config----------\n");
  fprintf(stdout, "encoder: %s\n", fire_red_asr->encoder);
  fprintf(stdout, "decoder: %s\n", fire_red_asr->decoder);

  fprintf(stdout, "----------offline Dolphin model config----------\n");
  fprintf(stdout, "model: %s\n", dolphin->model);

  fprintf(stdout, "----------offline zipformer ctc model config----------\n");
  fprintf(stdout, "model: %s\n", zipformer_ctc->model);

  fprintf(stdout, "----------offline NeMo Canary model config----------\n");
  fprintf(stdout, "encoder: %s\n", canary->encoder);
  fprintf(stdout, "decoder: %s\n", canary->decoder);
  fprintf(stdout, "src_lang: %s\n", canary->src_lang);
  fprintf(stdout, "tgt_lang: %s\n", canary->tgt_lang);
  fprintf(stdout, "use_pnc: %d\n", canary->use_pnc);

  fprintf(stdout, "----------offline wenet ctc model config----------\n");
  fprintf(stdout, "model: %s\n", wenet_ctc->model);

  fprintf(stdout, "----------offline Omnilingual ASR model config----------\n");
  fprintf(stdout, "model: %s\n", omnilingual->model);

  fprintf(stdout, "----------offline MedASR model config----------\n");
  fprintf(stdout, "model: %s\n", medasr->model);

  fprintf(stdout, "----------offline FunASR Nano config----------\n");
  fprintf(stdout, "encoder_adaptor: %s\n", funasr_nano->encoder_adaptor);
  fprintf(stdout, "llm: %s\n", funasr_nano->llm);
  fprintf(stdout, "embedding: %s\n", funasr_nano->embedding);
  fprintf(stdout, "tokenizer: %s\n", funasr_nano->tokenizer);
  fprintf(stdout, "system_prompt: %s\n", funasr_nano->system_prompt);
  fprintf(stdout, "user_prompt: %s\n", funasr_nano->user_prompt);
  fprintf(stdout, "max_new_tokens: %d\n", funasr_nano->max_new_tokens);
  fprintf(stdout, "temperature: %f\n", funasr_nano->temperature);
  fprintf(stdout, "top_p: %f\n", funasr_nano->top_p);
  fprintf(stdout, "seed: %d\n", funasr_nano->seed);

  fprintf(stdout, "----------offline FireRedASR CTC model config----------\n");
  fprintf(stdout, "model: %s\n", fire_red_asr_ctc->model);

  fprintf(stdout, "----------offline Qwen3 ASR model config----------\n");
  fprintf(stdout, "conv_frontend: %s\n", qwen3_asr->conv_frontend);
  fprintf(stdout, "encoder: %s\n", qwen3_asr->encoder);
  fprintf(stdout, "decoder: %s\n", qwen3_asr->decoder);
  fprintf(stdout, "tokenizer: %s\n", qwen3_asr->tokenizer);
  fprintf(stdout, "max_total_len: %d\n", qwen3_asr->max_total_len);
  fprintf(stdout, "max_new_tokens: %d\n", qwen3_asr->max_new_tokens);
  fprintf(stdout, "temperature: %f\n", qwen3_asr->temperature);
  fprintf(stdout, "top_p: %f\n", qwen3_asr->top_p);
  fprintf(stdout, "seed: %d\n", qwen3_asr->seed);
  fprintf(stdout, "hotwords: %s\n", qwen3_asr->hotwords);

  fprintf(stdout, "----------offline Cohere Transcribe model config----------\n");
  fprintf(stdout, "encoder: %s\n", cohere_transcribe->encoder);
  fprintf(stdout, "decoder: %s\n", cohere_transcribe->decoder);
  fprintf(stdout, "language: %s\n", cohere_transcribe->language);
  fprintf(stdout, "use_punct: %d\n", cohere_transcribe->use_punct);
  fprintf(stdout, "use_itn: %d\n", cohere_transcribe->use_itn);

  fprintf(stdout, "tokens: %s\n", model_config->tokens);
  fprintf(stdout, "num_threads: %d\n", model_config->num_threads);
  fprintf(stdout, "provider: %s\n", model_config->provider);
  fprintf(stdout, "debug: %d\n", model_config->debug);
  fprintf(stdout, "model type: %s\n", model_config->model_type);
  fprintf(stdout, "modeling unit: %s\n", model_config->modeling_unit);
  fprintf(stdout, "bpe vocab: %s\n", model_config->bpe_vocab);
  fprintf(stdout, "telespeech_ctc: %s\n", model_config->telespeech_ctc);

  fprintf(stdout, "----------feat config----------\n");
  fprintf(stdout, "sample rate: %d\n", feat->sample_rate);
  fprintf(stdout, "feat dim: %d\n", feat->feature_dim);

  fprintf(stdout, "----------recognizer config----------\n");
  fprintf(stdout, "decoding method: %s\n", config->decoding_method);
  fprintf(stdout, "max active paths: %d\n", config->max_active_paths);
  fprintf(stdout, "hotwords_file: %s\n", config->hotwords_file);
  fprintf(stdout, "hotwords_score: %.2f\n", config->hotwords_score);
  fprintf(stdout, "rule_fsts: %s\n", config->rule_fsts);
  fprintf(stdout, "rule_fars: %s\n", config->rule_fars);
  fprintf(stdout, "blank_penalty: %f\n", config->blank_penalty);
  fprintf(stdout, "----------hr config----------\n");
  fprintf(stdout, "dict_dir: %s\n", config->hr.dict_dir);
  fprintf(stdout, "lexicon: %s\n", config->hr.lexicon);
  fprintf(stdout, "rule_fsts: %s\n", config->hr.rule_fsts);
}

void CopyHeap(const char *src, int32_t num_bytes, char *dst) {
  std::copy(src, src + num_bytes, dst);
}
}
