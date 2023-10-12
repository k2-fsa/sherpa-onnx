// sherpa-onnx/csrc/sherpa-onnx-offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-tts-synthesizer.h"
#include "sherpa-onnx/csrc/parse-options.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline text-to-speech with sherpa-onnx

./bin/sherpa-onnx-offline-tts \
 --vits-model /path/to/model.onnx \
 --vits-lexicon /path/to/lexicon.txt \
 --vits-tokens /path/to/tokens.txt
 "some text within double quotes"

It will generate a file test.wav.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineTtsSynthesizerConfig config;
  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() != 1) {
    fprintf(stderr, "Error: Please provide the text to generate audio.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  return 0;
}
