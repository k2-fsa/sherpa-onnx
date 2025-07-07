// sherpa-onnx/csrc/offline-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-canary-model-config.h"
#include "sherpa-onnx/csrc/offline-dolphin-model-config.h"
#include "sherpa-onnx/csrc/offline-fire-red-asr-model-config.h"
#include "sherpa-onnx/csrc/offline-moonshine-model-config.h"
#include "sherpa-onnx/csrc/offline-nemo-enc-dec-ctc-model-config.h"
#include "sherpa-onnx/csrc/offline-paraformer-model-config.h"
#include "sherpa-onnx/csrc/offline-sense-voice-model-config.h"
#include "sherpa-onnx/csrc/offline-tdnn-model-config.h"
#include "sherpa-onnx/csrc/offline-transducer-model-config.h"
#include "sherpa-onnx/csrc/offline-wenet-ctc-model-config.h"
#include "sherpa-onnx/csrc/offline-whisper-model-config.h"
#include "sherpa-onnx/csrc/offline-zipformer-ctc-model-config.h"

namespace sherpa_onnx {

struct OfflineModelConfig {
  OfflineTransducerModelConfig transducer;
  OfflineParaformerModelConfig paraformer;
  OfflineNemoEncDecCtcModelConfig nemo_ctc;
  OfflineWhisperModelConfig whisper;
  OfflineFireRedAsrModelConfig fire_red_asr;
  OfflineTdnnModelConfig tdnn;
  OfflineZipformerCtcModelConfig zipformer_ctc;
  OfflineWenetCtcModelConfig wenet_ctc;
  OfflineSenseVoiceModelConfig sense_voice;
  OfflineMoonshineModelConfig moonshine;
  OfflineDolphinModelConfig dolphin;
  OfflineCanaryModelConfig canary;
  std::string telespeech_ctc;

  std::string tokens;
  int32_t num_threads = 2;
  bool debug = false;
  std::string provider = "cpu";

  // With the help of this field, we only need to load the model once
  // instead of twice; and therefore it reduces initialization time.
  //
  // Valid values:
  //  - transducer. The given model is from icefall
  //  - paraformer. It is a paraformer model
  //  - nemo_ctc. It is a NeMo CTC model.
  //
  // All other values are invalid and lead to loading the model twice.
  std::string model_type;

  std::string modeling_unit = "cjkchar";
  std::string bpe_vocab;

  OfflineModelConfig() = default;
  OfflineModelConfig(const OfflineTransducerModelConfig &transducer,
                     const OfflineParaformerModelConfig &paraformer,
                     const OfflineNemoEncDecCtcModelConfig &nemo_ctc,
                     const OfflineWhisperModelConfig &whisper,
                     const OfflineFireRedAsrModelConfig &fire_red_asr,
                     const OfflineTdnnModelConfig &tdnn,
                     const OfflineZipformerCtcModelConfig &zipformer_ctc,
                     const OfflineWenetCtcModelConfig &wenet_ctc,
                     const OfflineSenseVoiceModelConfig &sense_voice,
                     const OfflineMoonshineModelConfig &moonshine,
                     const OfflineDolphinModelConfig &dolphin,
                     const OfflineCanaryModelConfig &canary,
                     const std::string &telespeech_ctc,
                     const std::string &tokens, int32_t num_threads, bool debug,
                     const std::string &provider, const std::string &model_type,
                     const std::string &modeling_unit,
                     const std::string &bpe_vocab)
      : transducer(transducer),
        paraformer(paraformer),
        nemo_ctc(nemo_ctc),
        whisper(whisper),
        fire_red_asr(fire_red_asr),
        tdnn(tdnn),
        zipformer_ctc(zipformer_ctc),
        wenet_ctc(wenet_ctc),
        sense_voice(sense_voice),
        moonshine(moonshine),
        dolphin(dolphin),
        canary(canary),
        telespeech_ctc(telespeech_ctc),
        tokens(tokens),
        num_threads(num_threads),
        debug(debug),
        provider(provider),
        model_type(model_type),
        modeling_unit(modeling_unit),
        bpe_vocab(bpe_vocab) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_MODEL_CONFIG_H_
