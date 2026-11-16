// c-api-examples/forced-alignment-zh-c-api.c
//
// Copyright (c)  2026  Xiaomi Corporation

//
// This file demonstrates forced alignment of Chinese speech with
// sherpa-onnx's C API. It aligns the test wav of the zh-aishell3-pinyin
// model package with its transcript and writes a Praat TextGrid.
// clang-format off
/*
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
tar xvf zh-aishell3-pinyin.tar.bz2
rm zh-aishell3-pinyin.tar.bz2
*/
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  const char* model_dir = "./zh-aishell3-pinyin";
  const char* wav_filename = "./zh-aishell3-pinyin/zh.wav";

  // The transcript contains hanzi; the file must be saved as UTF-8.
  const char* text = "不要问你的国家能为你做什么，而要问你能为你的国家做什么。";

  const SherpaOnnxWave* wave = SherpaOnnxReadWave(wav_filename);
  if (!wave) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  char lexicon[256];
  snprintf(lexicon, sizeof(lexicon), "%s/lexicon.txt", model_dir);

  SherpaOnnxForcedAlignerConfig config;
  memset(&config, 0, sizeof(config));
  config.model_dir = model_dir;
  config.lexicon = lexicon;

  SherpaOnnxForcedAligner* aligner = SherpaOnnxCreateForcedAligner(&config);
  if (!aligner) {
    fprintf(stderr, "Failed to create the forced aligner!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  SherpaOnnxOfflineStream* stream =
      SherpaOnnxForcedAlignerCreateStream(aligner);

  // Optional: per-stream options, e.g. debug output of one utterance.
  SherpaOnnxOfflineStreamSetOption(stream, "debug", "1");

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);

  const SherpaOnnxForcedAlignmentResult* result =
      SherpaOnnxForcedAlignerCompute(aligner, stream, text);

  if (result) {
    printf("duration: %.3f seconds\n", result->duration);

    for (int32_t i = 0; i != result->num_tiers; ++i) {
      const SherpaOnnxForcedAlignmentTier* t = &result->tiers[i];
      printf("tier %s: %d intervals\n", t->name, t->num_intervals);
    }

    const char* textgrid_filename = "zh.TextGrid";
    FILE* fp = fopen(textgrid_filename, "wb");
    fprintf(fp, "%s", result->textgrid);
    fclose(fp);
    printf("\n%s created!\n", textgrid_filename);

    SherpaOnnxDestroyForcedAlignmentResult(result);
  }

  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxDestroyForcedAligner(aligner);
  SherpaOnnxFreeWave(wave);

  return 0;
}
