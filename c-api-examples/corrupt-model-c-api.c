// c-api-examples/corrupt-model-c-api.c
//
// Copyright (c)  2026  Cassian433

// This file checks that SherpaOnnxCreateXXX() returns NULL, instead of
// aborting the process, when a model file is corrupted, e.g., when it was
// only partially downloaded.
//
// You don't need to download any model files to run it. It writes a fake
// model file and a tokens file to the current directory and removes both
// before returning.

#include <stdio.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static int32_t WriteTextFile(const char *filename, const char *text) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "Failed to create %s\n", filename);
    return -1;
  }
  fputs(text, fp);
  fclose(fp);
  return 0;
}

static int32_t Check(const char *name, const void *p) {
  if (p) {
    fprintf(stderr, "%s: expected NULL for a corrupted model\n", name);
    return 1;
  }

  fprintf(stderr, "%s: NULL\n", name);
  return 0;
}

int32_t main() {
  const char *model = "./corrupt-model.onnx";
  const char *tokens = "./corrupt-model-tokens.txt";

  if (WriteTextFile(model, "This is not an onnx model\n") != 0 ||
      WriteTextFile(tokens, "<blk> 0\na 1\n") != 0) {
    return -1;
  }

  int32_t num_errors = 0;

  {
    SherpaOnnxOfflineRecognizerConfig config;
    memset(&config, 0, sizeof(config));
    config.model_config.sense_voice.model = model;
    config.model_config.tokens = tokens;
    config.model_config.num_threads = 1;

    const SherpaOnnxOfflineRecognizer *p =
        SherpaOnnxCreateOfflineRecognizer(&config);
    num_errors += Check("OfflineRecognizer", p);
    SherpaOnnxDestroyOfflineRecognizer(p);
  }

  {
    SherpaOnnxOnlineRecognizerConfig config;
    memset(&config, 0, sizeof(config));
    config.model_config.zipformer2_ctc.model = model;
    config.model_config.tokens = tokens;
    config.model_config.num_threads = 1;

    const SherpaOnnxOnlineRecognizer *p =
        SherpaOnnxCreateOnlineRecognizer(&config);
    num_errors += Check("OnlineRecognizer", p);
    SherpaOnnxDestroyOnlineRecognizer(p);
  }

  {
    SherpaOnnxKeywordSpotterConfig config;
    memset(&config, 0, sizeof(config));
    config.model_config.transducer.encoder = model;
    config.model_config.transducer.decoder = model;
    config.model_config.transducer.joiner = model;
    config.model_config.tokens = tokens;
    config.model_config.num_threads = 1;
    config.keywords_buf = "a";
    config.keywords_buf_size = 1;

    const SherpaOnnxKeywordSpotter *p = SherpaOnnxCreateKeywordSpotter(&config);
    num_errors += Check("KeywordSpotter", p);
    SherpaOnnxDestroyKeywordSpotter(p);
  }

  {
    SherpaOnnxVadModelConfig config;
    memset(&config, 0, sizeof(config));
    config.silero_vad.model = model;
    config.num_threads = 1;

    const SherpaOnnxVoiceActivityDetector *p =
        SherpaOnnxCreateVoiceActivityDetector(&config, 30);
    num_errors += Check("VoiceActivityDetector", p);
    SherpaOnnxDestroyVoiceActivityDetector(p);
  }

  {
    SherpaOnnxOfflineTtsConfig config;
    memset(&config, 0, sizeof(config));
    config.model.vits.model = model;
    config.model.vits.tokens = tokens;
    config.model.num_threads = 1;

    const SherpaOnnxOfflineTts *p = SherpaOnnxCreateOfflineTts(&config);
    num_errors += Check("OfflineTts", p);
    SherpaOnnxDestroyOfflineTts(p);
  }

  {
    SherpaOnnxSpeakerEmbeddingExtractorConfig config;
    memset(&config, 0, sizeof(config));
    config.model = model;
    config.num_threads = 1;

    const SherpaOnnxSpeakerEmbeddingExtractor *p =
        SherpaOnnxCreateSpeakerEmbeddingExtractor(&config);
    num_errors += Check("SpeakerEmbeddingExtractor", p);
    SherpaOnnxDestroySpeakerEmbeddingExtractor(p);
  }

  {
    SherpaOnnxOfflinePunctuationConfig config;
    memset(&config, 0, sizeof(config));
    config.model.ct_transformer = model;
    config.model.num_threads = 1;

    const SherpaOnnxOfflinePunctuation *p =
        SherpaOnnxCreateOfflinePunctuation(&config);
    num_errors += Check("OfflinePunctuation", p);
    SherpaOnnxDestroyOfflinePunctuation(p);
  }

  remove(model);
  remove(tokens);

  return num_errors == 0 ? 0 : -1;
}
