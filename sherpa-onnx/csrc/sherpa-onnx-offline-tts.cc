// sherpa-onnx/csrc/sherpa-onnx-offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <fstream>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-writer.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline text-to-speech with sherpa-onnx

./bin/sherpa-onnx-offline-tts \
 --vits-model /path/to/model.onnx \
 --vits-lexicon /path/to/lexicon.txt \
 --vits-tokens /path/to/tokens.txt
 --output-filename ./generated.wav \
 'some text within single quotes'

It will generate a file ./generated.wav as specified by --output-filename.

You can download a test model from
https://huggingface.co/csukuangfj/vits-ljs

For instance, you can use:
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
wget https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt

./bin/sherpa-onnx-offline-tts \
  --vits-model=./vits-ljs.onnx \
  --vits-lexicon=./lexicon.txt \
  --vits-tokens=./tokens.txt \
  --output-filename=./generated.wav \
  'liliana, the most beautiful and lovely assistant of our team!'
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  std::string output_filename = "./generated.wav";
  po.Register("output-filename", &output_filename,
              "Path to save the generated audio");

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

  bool ok = sherpa_onnx::WriteWave(output_filename, audio.sample_rate,
                                   audio.samples.data(), audio.samples.size());
  if (!ok) {
    fprintf(stderr, "Failed to write wave to %s\n", output_filename.c_str());
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "The text is: %s\n", po.GetArg(1).c_str());
  fprintf(stderr, "Saved to %s successfully!\n", output_filename.c_str());

  return 0;
}
