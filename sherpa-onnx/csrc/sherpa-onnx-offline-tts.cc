// sherpa-onnx/csrc/sherpa-onnx-offline-tts.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include <chrono>  // NOLINT
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>

#include "sherpa-onnx/csrc/offline-tts.h"
#include "sherpa-onnx/csrc/macros.h"
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
Offline/Non-streaming text-to-speech with sherpa-onnx

Usage examples:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

./bin/sherpa-onnx-offline-tts \
 --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
 --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt \
 --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
 --output-filename=./generated.wav \
  "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

Pocket TTS:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2
tar xf sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

./bin/sherpa-onnx-offline-tts \
 --pocket-lm-flow=./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx \
 --pocket-lm-main=./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx \
 --pocket-encoder=./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx \
 --pocket-decoder=./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx \
 --pocket-text-conditioner=./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx \
 --pocket-vocab-json=./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json \
 --pocket-token-scores-json=./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json \
 --reference-audio=./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav \
 --output-filename=./generated-pocket.wav \
 "Hello from Pocket TTS"

Supertonic TTS:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2
tar xf sherpa-onnx-supertonic-tts-int8-2026-03-06.tar.bz2

./bin/sherpa-onnx-offline-tts \
 --supertonic-duration-predictor=./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx \
 --supertonic-text-encoder=./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx \
 --supertonic-vector-estimator=./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx \
 --supertonic-vocoder=./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx \
 --supertonic-tts-json=./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json \
 --supertonic-unicode-indexer=./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin \
 --supertonic-voice-style=./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin \
 --lang=en \
 --output-filename=./generated-supertonic.wav \
 "Hello from Supertonic TTS"

ZipVoice TTS:

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
tar xf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

./bin/sherpa-onnx-offline-tts \
 --zipvoice-encoder=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx \
 --zipvoice-decoder=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx \
 --zipvoice-data-dir=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data \
 --zipvoice-lexicon=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt \
 --zipvoice-tokens=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt \
 --zipvoice-vocoder=./vocos_24khz.onnx \
 --reference-audio=./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav \
 --reference-text="那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系." \
 --num-steps=4 \
 --output-filename=./generated-zipvoice.wav \
 "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."

It will generate a file specified by --output-filename.

You can find more models at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models

Please see
https://k2-fsa.github.io/sherpa/onnx/tts/index.html
or details.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  std::string output_filename = "./generated.wav";
  int32_t sid = 0;

  std::string reference_audio;
  po.Register(
      "reference-audio", &reference_audio,
      "Path to reference audio. Required by Pocket TTS and ZipVoice TTS.");

  std::string reference_text;
  po.Register(
      "reference-text", &reference_text,
      "Reference text for the reference audio. Required by ZipVoice TTS.");

  sherpa_onnx::GenerationConfig gen_config;

  std::string lang;

  po.Register(
      "num-steps", &gen_config.num_steps,
      "Used by some models, e.g., Pocket TTS and ZipVoice. Number of flow "
      "matching steps.");

  po.Register("output-filename", &output_filename,
              "Path to save the generated audio");

  po.Register("lang", &lang,
              "Language for text: en, ko, es, pt, fr. Used only by "
              "Supertonic TTS.");

  po.Register("sid", &sid,
              "Speaker ID. Used only for multi-speaker models, e.g., models "
              "trained using the VCTK dataset. Not used for single-speaker "
              "models, e.g., models trained using the LJSpeech dataset");

  po.Register("speed", &gen_config.speed,
              "Speech speed. Larger=faster. Used by Supertonic, VITS, etc. "
              "(float, default = 1.0)");

  sherpa_onnx::OfflineTtsConfig config;

  config.Register(&po);
  po.Read(argc, argv);

  if (po.NumArgs() == 0) {
    fprintf(stderr, "Error: Please provide the text to generate audio.\n\n");
    po.PrintUsage();
    SHERPA_ONNX_EXIT(EXIT_FAILURE);
  }

  if (po.NumArgs() > 1) {
    fprintf(stderr,
            "Error: Accept only one positional argument. Please use single "
            "quotes to wrap your text.\n");
    po.PrintUsage();
    SHERPA_ONNX_EXIT(EXIT_FAILURE);
  }

  if (config.model.debug) {
    fprintf(stderr, "%s\n", config.model.ToString().c_str());
  }

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    SHERPA_ONNX_EXIT(EXIT_FAILURE);
  }

  sherpa_onnx::OfflineTts tts(config);

  const auto begin = std::chrono::steady_clock::now();
  sherpa_onnx::GeneratedAudio audio;

  bool is_pocket_tts = !config.model.pocket.lm_flow.empty();
  bool is_supertonic_tts = !config.model.supertonic.tts_json.empty();
  bool is_zipvoice_tts = !config.model.zipvoice.encoder.empty() &&
                         !config.model.zipvoice.decoder.empty();

  gen_config.sid = sid;

  if (is_supertonic_tts && !lang.empty()) {
    gen_config.extra["lang"] = lang;
  }

  if (is_pocket_tts || is_zipvoice_tts) {
    if (reference_audio.empty()) {
      fprintf(stderr,
              "You need to provide --reference-audio for this TTS model");
      SHERPA_ONNX_EXIT(EXIT_FAILURE);
    }

    int32_t sample_rate;
    bool is_ok = false;
    auto samples =
        sherpa_onnx::ReadWave(reference_audio, &sample_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'", reference_audio.c_str());
      SHERPA_ONNX_EXIT(EXIT_FAILURE);
    }

    gen_config.reference_audio = std::move(samples);
    gen_config.reference_sample_rate = sample_rate;
  }

  if (is_zipvoice_tts) {
    if (reference_text.empty()) {
      fprintf(stderr,
              "You need to provide --reference-text for ZipVoice TTS");
      SHERPA_ONNX_EXIT(EXIT_FAILURE);
    }
    gen_config.reference_text = reference_text;
  }

  audio = tts.Generate(po.GetArg(1), gen_config, AudioCallback);

  const auto end = std::chrono::steady_clock::now();

  if (audio.samples.empty()) {
    fprintf(
        stderr,
        "Error in generating audio. Please read previous error messages.\n");
    SHERPA_ONNX_EXIT(EXIT_FAILURE);
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  float duration = audio.samples.size() / static_cast<float>(audio.sample_rate);

  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Number of threads: %d\n", config.model.num_threads);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  fprintf(stderr, "Audio duration: %.3f s\n", duration);
  fprintf(stderr, "Real-time factor (RTF): %.3f/%.3f = %.3f\n", elapsed_seconds,
          duration, rtf);

  bool ok = sherpa_onnx::WriteWave(output_filename, audio.sample_rate,
                                   audio.samples.data(), audio.samples.size());
  if (!ok) {
    fprintf(stderr, "Failed to write wave to %s\n", output_filename.c_str());
    SHERPA_ONNX_EXIT(EXIT_FAILURE);
  }

  fprintf(stderr, "The text is: %s. Speaker ID: %d\n", po.GetArg(1).c_str(),
          sid);
  fprintf(stderr, "Saved to %s successfully!\n", output_filename.c_str());

  return 0;
}
