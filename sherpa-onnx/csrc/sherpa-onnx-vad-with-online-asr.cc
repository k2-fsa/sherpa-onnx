// sherpa-onnx/csrc/sherpa-onnx-vad-with-online-asr.cc
//
// Copyright (c)  2025  Xiaomi Corporation
// Copyright (c)  2025  Pingfeng Luo
//
// This file demonstrates how to use vad in streaming speech recognition
//

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/resample.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/voice-activity-detector.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int32_t main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Speech recognition using VAD + streaming models with sherpa-onnx-vad-with-online-asr.
This is useful when testing long audio.

Usage:

Note you can download silero_vad.onnx using

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

(1) Streaming transducer

  ./bin/sherpa-onnx-vad-with-online-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=2 \
    --decoding-method=greedy_search \
    /path/to/long_duration.wav

(2) Streaming zipformer2 CTC

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2

  ./bin/sherpa-onnx-vad-with-online-asr \
    --debug=1 \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --zipformer2-ctc-model=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/ctc-epoch-20-avg-1-chunk-16-left-128.onnx \
    --tokens=./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/tokens.txt \
    ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13/test_wavs/DEV_T0000000000.wav

(3) Streaming paraformer

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
  tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

  ./bin/sherpa-onnx-vad-with-online-asr \
    --silero-vad-model=/path/to/silero_vad.onnx \
    --tokens=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt \
    --paraformer-encoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.onnx \
    --paraformer-decoder=./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.onnx \
    /path/to/long_duration.wav


The input wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OnlineRecognizerConfig asr_config;
  asr_config.Register(&po);

  sherpa_onnx::VadModelConfig vad_config;
  vad_config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr, "Error: Please provide exactly 1 wave file. Given: %d\n\n",
            po.NumArgs());
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", vad_config.ToString().c_str());
  fprintf(stderr, "%s\n", asr_config.ToString().c_str());

  if (!vad_config.Validate()) {
    fprintf(stderr, "Errors in vad_config!\n");
    return -1;
  }

  if (!asr_config.Validate()) {
    fprintf(stderr, "Errors in ASR config!\n");
    return -1;
  }

  fprintf(stderr, "Creating recognizer ...\n");
  sherpa_onnx::OnlineRecognizer recognizer(asr_config);
  fprintf(stderr, "Recognizer created!\n");

  auto vad = std::make_unique<sherpa_onnx::VoiceActivityDetector>(vad_config);

  fprintf(stderr, "Started\n");
  const auto begin = std::chrono::steady_clock::now();

  std::string wave_filename = po.GetArg(1);
  fprintf(stderr, "Reading: %s\n", wave_filename.c_str());
  int32_t sampling_rate = -1;
  bool is_ok = false;
  auto samples = sherpa_onnx::ReadWave(wave_filename, &sampling_rate, &is_ok);
  if (!is_ok) {
    fprintf(stderr, "Failed to read '%s'\n", wave_filename.c_str());
    return -1;
  }

  if (sampling_rate != 16000) {
    fprintf(stderr, "Resampling from %d Hz to 16000 Hz\n", sampling_rate);
    float min_freq = std::min(sampling_rate, 16000)
    float lowpass_cutoff = 0.99 * 0.5 * min_freq;

    int32_t lowpass_filter_width = 6;
    auto resampler = std::make_unique<sherpa_onnx::LinearResample>(
        sampling_rate, 16000, lowpass_cutoff, lowpass_filter_width);
    std::vector<float> out_samples;
    resampler->Resample(samples.data(), samples.size(), true, &out_samples);
    samples = std::move(out_samples);
    fprintf(stderr, "Resampling done\n");
  }

  fprintf(stderr, "Started!\n");
  int32_t window_size = vad_config.ten_vad.model.empty()
    ? vad_config.silero_vad.window_size : vad_config.ten_vad.window_size;
  int32_t offset = 0;
  int32_t segment_id = 0;
  bool speech_started = false;
  while (offset < samples.size()) {
    if (offset + window_size <= samples.size()) {
      vad->AcceptWaveform(samples.data() + offset, window_size);
    } else {
      vad->Flush();
    }
    offset += window_size;
    if (vad->IsSpeechDetected() && !speech_started) {
      // new voice activity
      speech_started = true;
      segment_id++;
    } else if (!vad->IsSpeechDetected() && speech_started) {
      // end voice activity
      speech_started = false;
    }

    while (!vad->Empty()) {
      const auto &segment = vad->Front();
      float duration = segment.samples.size() / 16000.;
      float start_time = segment.start / 16000.;
      float end_time = start_time + duration;
      auto s = recognizer.CreateStream();
      s->AcceptWaveform(16000, segment.samples.data(), segment.samples.size());
      s->InputFinished();
      while (recognizer.IsReady(s.get())) {
        recognizer.DecodeStream(s.get());
      }
      auto text = recognizer.GetResult(s.get()).text;
      if (!text.empty()) {
        fprintf(stderr, "vad segment(%d:%.3f-%.3f) results: %s\n",
            segment_id, start_time, end_time, text.c_str());
      }
      vad->Pop();
    }
  }

  const auto end = std::chrono::steady_clock::now();

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "num threads: %d\n", asr_config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", asr_config.decoding_method.c_str());
  if (asr_config.decoding_method == "modified_beam_search") {
    fprintf(stderr, "max active paths: %d\n", asr_config.max_active_paths);
  }

  float duration = samples.size() / 16000.;
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, duration, rtf);

  return 0;
}
