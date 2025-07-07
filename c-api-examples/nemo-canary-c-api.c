// c-api-examples/nemo-canary-c-api.c
//
// Copyright (c)  2025  Xiaomi Corporation

// We assume you have pre-downloaded the Nemo Canary model
// from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
// An example is given below:
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
// tar xvf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
// rm sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
//
// clang-format on
//
// see https://k2-fsa.github.io/sherpa/onnx/nemo/canary.html
// for details

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char *wav_filename =
      "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/de.wav";
  const char *encoder_filename =
      "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx";
  const char *decoder_filename =
      "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx";
  const char *tokens_filename =
      "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt";
  const char *provider = "cpu";

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  // Offline model config
  SherpaOnnxOfflineModelConfig offline_model_config;
  memset(&offline_model_config, 0, sizeof(offline_model_config));

  // set debug to 1 to view more logs
  offline_model_config.debug = 0;

  offline_model_config.num_threads = 1;
  offline_model_config.provider = provider;
  offline_model_config.tokens = tokens_filename;
  offline_model_config.canary.encoder = encoder_filename;
  offline_model_config.canary.decoder = decoder_filename;

  // so it output punctuations and cases
  offline_model_config.canary.use_pnc = 1;

  offline_model_config.canary.src_lang = "de";

  // since there is a German audio, you can set tgt_lang to en or de
  offline_model_config.canary.tgt_lang = "en";

  // Recognizer config
  SherpaOnnxOfflineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.decoding_method = "greedy_search";
  recognizer_config.model_config = offline_model_config;

  const SherpaOnnxOfflineRecognizer *recognizer =
      SherpaOnnxCreateOfflineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");

    SherpaOnnxFreeWave(wave);

    return -1;
  }

  const SherpaOnnxOfflineStream *stream =
      SherpaOnnxCreateOfflineStream(recognizer);

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);
  SherpaOnnxDecodeOfflineStream(recognizer, stream);
  const SherpaOnnxOfflineRecognizerResult *result =
      SherpaOnnxGetOfflineStreamResult(stream);

  fprintf(stderr, "Decoded text (English): %s\n", result->text);

  SherpaOnnxDestroyOfflineRecognizerResult(result);
  SherpaOnnxDestroyOfflineStream(stream);

  // now output German text
  recognizer_config.model_config.canary.tgt_lang = "de";
  SherpaOnnxOfflineRecognizerSetConfig(recognizer, &recognizer_config);

  stream = SherpaOnnxCreateOfflineStream(recognizer);

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);
  SherpaOnnxDecodeOfflineStream(recognizer, stream);
  result = SherpaOnnxGetOfflineStreamResult(stream);

  fprintf(stderr, "Decoded text (German): %s\n", result->text);

  SherpaOnnxDestroyOfflineRecognizerResult(result);
  SherpaOnnxDestroyOfflineStream(stream);

  SherpaOnnxDestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return 0;
}
