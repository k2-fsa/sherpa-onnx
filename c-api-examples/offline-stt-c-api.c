// c-api-examples/offline-stt-c-api.c
//
// Copyright (c)  2024  Xiaomi Corporation

// We assume you have pre-downloaded the whisper multi-lingual models
// from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
// An example command to download the "tiny" whisper model is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
// tar xvf sherpa-onnx-whisper-tiny.tar.bz2
// rm sherpa-onnx-whisper-tiny.tar.bz2
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  // You can find more test waves from
  // https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/tree/main/test_wavs
  const char *wav_filename = "./sherpa-onnx-whisper-tiny/test_wavs/0.wav";
  const char *encoder_filename = "sherpa-onnx-whisper-tiny/tiny-encoder.onnx";
  const char *decoder_filename = "sherpa-onnx-whisper-tiny/tiny-decoder.onnx";
  const char *tokens_filename = "sherpa-onnx-whisper-tiny/tiny-tokens.txt";
  const char *language = "en";
  const char *provider = "cpu";

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // Whisper config
  SherpaOnnxOfflineWhisperModelConfig whisper_config;
  whisper_config.decoder = decoder_filename;
  whisper_config.encoder = encoder_filename;
  whisper_config.language = language;
  whisper_config.tail_paddings = 0;
  whisper_config.task = "transcribe";

  // Offline model config
  SherpaOnnxOfflineModelConfig offline_model_config;
  offline_model_config.bpe_vocab = "";
  offline_model_config.debug = 1;
  offline_model_config.model_type = NULL;
  offline_model_config.modeling_unit = NULL;
  offline_model_config.nemo_ctc =
      (SherpaOnnxOfflineNemoEncDecCtcModelConfig){NULL};
  offline_model_config.num_threads = 1;
  offline_model_config.paraformer =
      (SherpaOnnxOfflineParaformerModelConfig){NULL};
  offline_model_config.provider = provider;
  offline_model_config.tdnn = (SherpaOnnxOfflineTdnnModelConfig){NULL};
  offline_model_config.telespeech_ctc = NULL;
  offline_model_config.tokens = tokens_filename;
  offline_model_config.transducer =
      (SherpaOnnxOfflineTransducerModelConfig){NULL, NULL, NULL};
  offline_model_config.whisper = whisper_config;
  offline_model_config.sense_voice =
      (SherpaOnnxOfflineSenseVoiceModelConfig){"", "", 0};

  // Recognizer config
  SherpaOnnxOfflineRecognizerConfig recognizer_config;
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.feat_config = (SherpaOnnxFeatureConfig){16000, 512};
  recognizer_config.hotwords_file = NULL;
  recognizer_config.hotwords_score = 0.0;
  recognizer_config.lm_config = (SherpaOnnxOfflineLMConfig){NULL, 0.0};
  recognizer_config.max_active_paths = 0;
  recognizer_config.model_config = offline_model_config;
  recognizer_config.rule_fars = NULL;
  recognizer_config.rule_fsts = NULL;

  SherpaOnnxOfflineRecognizer *recognizer =
      CreateOfflineRecognizer(&recognizer_config);

  SherpaOnnxOfflineStream *stream = CreateOfflineStream(recognizer);

  AcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                        wave->num_samples);
  DecodeOfflineStream(recognizer, stream);
  SherpaOnnxOfflineRecognizerResult *result = GetOfflineStreamResult(stream);

  fprintf(stderr, "Decoded text: %s\n", result->text);

  DestroyOfflineRecognizerResult(result);
  DestroyOfflineStream(stream);
  DestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return 0;
}
