// sherpa-onnx/csrc/sherpa-onnx-fa.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include <chrono>  // NOLINT
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/forced-aligner.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

static const char kUsageMessage[] = R"usage(
Forced alignment with GMM acoustic models (meta.json/final.alimdl/tree).

Given a wave file and its transcript, this tool computes the start/end time
of every phone and word and writes a Praat TextGrid.

Model packages (each with its own test wav) are at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
see scripts/mfa/README.md for details.

Usage (Chinese):

  sherpa-onnx-fa \
    --model-dir=zh-aishell3-pinyin \
    --lexicon=zh-aishell3-pinyin/lexicon.txt \
    --wav=zh-aishell3-pinyin/zh.wav \
    --text="不要问你的国家能为你做什么，而要问你能为你的国家做什么。" \
    --out=zh.TextGrid

Usage (English):

  sherpa-onnx-fa \
    --model-dir=english-mfa-v3.1.0 \
    --lexicon=english-mfa-v3.1.0/lexicon.txt \
    --wav=english-mfa-v3.1.0/en.wav \
    --text="ask not what your country can do for you, ask what you can do for your country." \
    --out=en.TextGrid
)usage";

int32_t main(int32_t argc, char *argv[]) {
  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::ForcedAlignerConfig config;
  config.Register(&po);

  std::string wav;
  std::string text;
  std::string text_file;
  std::string out;
  po.Register("wav", &wav,
              "Path to the input wave file. Also accepted as "
              "the first positional argument.");
  po.Register("text", &text, "Transcript of the whole utterance");
  po.Register("text-file", &text_file,
              "Read the transcript from this file instead of --text");
  po.Register("out", &out,
              "Where to write the output Praat TextGrid. Prints to stdout "
              "if empty.");

  po.Read(argc, argv);

  if (wav.empty() && po.NumArgs() >= 1) {
    wav = po.GetArg(1);
  }
  if (wav.empty()) {
    fprintf(stderr, "Error: please provide --wav\n\n");
    po.PrintUsage();
    return 1;
  }
  if (text.empty() && !text_file.empty()) {
    std::ifstream is = sherpa_onnx::OpenInputFile(text_file);
    std::ostringstream ss;
    ss << is.rdbuf();
    text = ss.str();
    while (!text.empty() &&
           (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
      text.pop_back();
    }
  }
  if (text.empty()) {
    fprintf(stderr, "Error: please provide --text (or --text-file)\n\n");
    po.PrintUsage();
    return 1;
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return 1;
  }

  int32_t sampling_rate = 0;
  bool is_ok = false;
  std::vector<float> samples =
      sherpa_onnx::ReadWave(wav, &sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Error: failed to read '%s'\n", wav.c_str());
    return 1;
  }

  sherpa_onnx::ForcedAlignmentResult result;
  std::chrono::steady_clock::time_point begin_init;
  std::chrono::steady_clock::time_point end_init;
  std::chrono::steady_clock::time_point begin;
  std::chrono::steady_clock::time_point end;
  try {
    begin_init = std::chrono::steady_clock::now();
    sherpa_onnx::ForcedAligner aligner(config);
    end_init = std::chrono::steady_clock::now();

    begin = std::chrono::steady_clock::now();

    auto stream = aligner.CreateStream();

    stream->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    result = aligner.Compute(stream.get(), text);

    end = std::chrono::steady_clock::now();
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }

  if (result.tiers.empty()) {
    fprintf(stderr,
            "Error: empty result (the audio produced no feature frames)\n");
    return 1;
  }

  std::string textgrid = result.ToTextGrid();
  if (out.empty()) {
    fputs(textgrid.c_str(), stdout);
  } else {
    std::ofstream os = sherpa_onnx::OpenOutputFile(out);
    if (!os) {
      fprintf(stderr, "Error: failed to write '%s'\n", out.c_str());
      return 1;
    }
    os << textgrid;
    os.close();
    if (!os) {
      fprintf(stderr, "Error: failed to write '%s'\n", out.c_str());
      return 1;
    }
    fprintf(stderr, "TextGrid written to '%s'\n", out.c_str());
  }

  float init_seconds =
      std::chrono::duration<double>(end_init - begin_init).count();
  float elapsed_seconds = std::chrono::duration<double>(end - begin).count();
  float audio_duration = samples.size() / static_cast<float>(sampling_rate);

  fprintf(stderr, "Init is done in %.3f s\n", init_seconds);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, audio_duration, elapsed_seconds / audio_duration);
  fprintf(stderr, "phones: %zu, words: %zu\n",
          result.tiers.empty() ? 0 : result.tiers[0].intervals.size(),
          result.tiers.size() < 2 ? 0 : result.tiers[1].intervals.size());

  return 0;
}
