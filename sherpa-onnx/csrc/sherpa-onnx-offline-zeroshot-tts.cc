// sherpa-onnx/csrc/sherpa-onnx-offline-zeroshot-tts.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include <chrono>  // NOLINT
#include <fstream>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"
#include "sherpa-onnx/csrc/wave-writer.h"

static int32_t AudioCallback(const float * /*samples*/, int32_t n,
                             float progress) {
  printf("sample=%d, progress=%f\n", n, progress);
  return 1;
}

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Offline/Non-streaming zero-shot text-to-speech with sherpa-onnx

Usage example:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-zh-en-emilia.tar.bz2

./bin/sherpa-onnx-offline-zeroshot-tts \
  --zipvoice-flow-matching-model=sherpa-onnx-zipvoice-distill-zh-en-emilia/fm_decoder.onnx \
  --zipvoice-text-model=sherpa-onnx-zipvoice-distill-zh-en-emilia/text_encoder.onnx \
  --zipvoice-data-dir=sherpa-onnx-zipvoice-distill-zh-en-emilia/espeak-ng-data \
  --zipvoice-pinyin-dict=sherpa-onnx-zipvoice-distill-zh-en-emilia/pinyin.raw \
  --zipvoice-tokens=sherpa-onnx-zipvoice-distill-zh-en-emilia/tokens.txt \
  --zipvoice-vocoder=sherpa-onnx-zipvoice-distill-zh-en-emilia/vocos_24khz.onnx \
  --prompt-audio=sherpa-onnx-zipvoice-distill-zh-en-emilia/prompt.wav \
  --num-steps=4 \
  --num-threads=4 \
  --prompt-text="周日被我射熄火了，所以今天是周一。" \
  "我是中国人民的儿子，我爱我的祖国。我得祖国是一个伟大的国家，拥有五千年的文明史。"

It will generate a file ./generated.wav as specified by --output-filename.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  std::string output_filename = "./generated.wav";

  int32_t num_steps = 4;
  float speed = 1.0;
  std::string prompt_text;
  std::string prompt_audio;

  po.Register("output-filename", &output_filename,
              "Path to save the generated audio");

  po.Register("num-steps", &num_steps,
              "Number of inference steps for ZipVoice (default: 4)");

  po.Register("speed", &speed,
              "Speech speed for ZipVoice (default: 1.0, larger=faster, "
              "smaller=slower)");

  po.Register("prompt-text", &prompt_text, "The transcribe of prompt_samples.");

  po.Register("prompt-audio", &prompt_audio,
              "The prompt audio file, single channel pcm. ");

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

  if (config.model.debug) {
    fprintf(stderr, "%s\n", config.model.ToString().c_str());
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    exit(EXIT_FAILURE);
  }

  if (prompt_text.empty() || prompt_audio.empty()) {
    fprintf(stderr, "Please provide either --prompt-text and --prompt-audio\n");
    exit(EXIT_FAILURE);
  }

  sherpa_onnx::OfflineTts tts(config);

  int32_t sample_rate = -1;
  bool is_ok = false;
  const std::vector<float> prompt_samples =
      sherpa_onnx::ReadWave(prompt_audio, &sample_rate, &is_ok);

  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", prompt_audio.c_str());
    return -1;
  }

  const auto begin = std::chrono::steady_clock::now();
  auto audio = tts.Generate(po.GetArg(1), prompt_text, prompt_samples,
                            sample_rate, speed, num_steps, AudioCallback);
  const auto end = std::chrono::steady_clock::now();

  if (audio.samples.empty()) {
    fprintf(
        stderr,
        "Error in generating audio. Please read previous error messages.\n");
    exit(EXIT_FAILURE);
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = audio.samples.size() / static_cast<float>(audio.sample_rate);

  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  fprintf(stderr, "Audio duration: %.3f s\n", duration);
  fprintf(stderr, "Real-time factor (RTF): %.3f/%.3f = %.3f\n", elapsed_seconds,
          duration, rtf);

  bool ok = sherpa_onnx::WriteWave(output_filename, audio.sample_rate,
                                   audio.samples.data(), audio.samples.size());
  if (!ok) {
    fprintf(stderr, "Failed to write wave to %s\n", output_filename.c_str());
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "The text is: %s.\n", po.GetArg(1).c_str());
  fprintf(stderr, "Saved to %s successfully!\n", output_filename.c_str());

  return 0;
}
