// sherpa-onnx/csrc/sherpa-onnx-offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <fstream>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/parse-options.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline text-to-speech with sherpa-onnx

./bin/sherpa-onnx-offline-tts \
 --vits-model /path/to/model.onnx \
 --vits-lexicon /path/to/lexicon.txt \
 --vits-tokens /path/to/tokens.txt
 'some text within single quotes'

It will generate a file test.wav.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineTtsConfig config;
  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() == 0) {
    fprintf(stderr, "Error: Please provide the text to generate audio.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (po.NumArgs() > 1) {
    fprintf(stderr,
            "Error: Accept only one positional argument. Please use single "
            "quotes to wrap your text\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    exit(EXIT_FAILURE);
  }

  sherpa_onnx::OfflineTts tts(config);
  auto audio = tts.Generate(po.GetArg(1));

  std::ofstream os("t.pcm", std::ios::binary);
  os.write(reinterpret_cast<const char *>(audio.samples.data()),
           sizeof(float) * audio.samples.size());

  // sox -t raw -r 22050 -b 32 -e floating-point -c 1 ./t.pcm ./t.wav

  return 0;
}
