// sherpa-onnx/python/csrc/sherpa-onnx.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/sherpa-onnx.h"

#include "sherpa-onnx/python/csrc/alsa.h"
#include "sherpa-onnx/python/csrc/audio-tagging.h"
#include "sherpa-onnx/python/csrc/circular-buffer.h"
#include "sherpa-onnx/python/csrc/display.h"
#include "sherpa-onnx/python/csrc/endpoint.h"
#include "sherpa-onnx/python/csrc/features.h"
#include "sherpa-onnx/python/csrc/homophone-replacer.h"
#include "sherpa-onnx/python/csrc/keyword-spotter.h"
#include "sherpa-onnx/python/csrc/offline-ctc-fst-decoder-config.h"
#include "sherpa-onnx/python/csrc/offline-lm-config.h"
#include "sherpa-onnx/python/csrc/offline-model-config.h"
#include "sherpa-onnx/python/csrc/offline-punctuation.h"
#include "sherpa-onnx/python/csrc/offline-recognizer.h"
#include "sherpa-onnx/python/csrc/offline-source-separation.h"
#include "sherpa-onnx/python/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/python/csrc/offline-stream.h"
#include "sherpa-onnx/python/csrc/online-ctc-fst-decoder-config.h"
#include "sherpa-onnx/python/csrc/online-lm-config.h"
#include "sherpa-onnx/python/csrc/online-model-config.h"
#include "sherpa-onnx/python/csrc/online-punctuation.h"
#include "sherpa-onnx/python/csrc/online-recognizer.h"
#include "sherpa-onnx/python/csrc/online-stream.h"
#include "sherpa-onnx/python/csrc/speaker-embedding-extractor.h"
#include "sherpa-onnx/python/csrc/speaker-embedding-manager.h"
#include "sherpa-onnx/python/csrc/spoken-language-identification.h"
#include "sherpa-onnx/python/csrc/vad-model-config.h"
#include "sherpa-onnx/python/csrc/vad-model.h"
#include "sherpa-onnx/python/csrc/version.h"
#include "sherpa-onnx/python/csrc/voice-activity-detector.h"
#include "sherpa-onnx/python/csrc/wave-writer.h"

#if SHERPA_ONNX_ENABLE_TTS == 1
#include "sherpa-onnx/python/csrc/offline-tts.h"
#endif

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
#include "sherpa-onnx/python/csrc/fast-clustering.h"
#include "sherpa-onnx/python/csrc/offline-speaker-diarization-result.h"
#include "sherpa-onnx/python/csrc/offline-speaker-diarization.h"
#endif

namespace sherpa_onnx {

PYBIND11_MODULE(_sherpa_onnx, m) {
  m.doc() = "pybind11 binding of sherpa-onnx";

  PybindWaveWriter(&m);
  PybindAudioTagging(&m);
  PybindOfflinePunctuation(&m);
  PybindOnlinePunctuation(&m);
  PybindHomophoneReplacer(&m);

  PybindFeatures(&m);
  PybindOnlineCtcFstDecoderConfig(&m);
  PybindOnlineModelConfig(&m);
  PybindOnlineLMConfig(&m);
  PybindOnlineStream(&m);
  PybindEndpoint(&m);
  PybindOnlineRecognizer(&m);
  PybindKeywordSpotter(&m);
  PybindDisplay(&m);

  PybindOfflineStream(&m);
  PybindOfflineLMConfig(&m);
  PybindOfflineModelConfig(&m);
  PybindOfflineCtcFstDecoderConfig(&m);
  PybindOfflineRecognizer(&m);

  PybindVadModelConfig(&m);
  PybindVadModel(&m);
  PybindCircularBuffer(&m);
  PybindVoiceActivityDetector(&m);

#if SHERPA_ONNX_ENABLE_TTS == 1
  PybindOfflineTts(&m);
#else
  /* Define "empty" TTS sybmbols */
  m.attr("OfflineTtsKokoroModelConfig") = py::none();
  m.attr("OfflineTtsMatchaModelConfig") = py::none();
  m.attr("OfflineTtsModelConfig") = py::none();
  m.attr("OfflineTtsVitsModelConfig") = py::none();
  m.attr("GeneratedAudio") = py::none();
  m.attr("OfflineTtsConfig") = py::none();
  m.attr("OfflineTts") = py::none();
#endif

  PybindSpeakerEmbeddingExtractor(&m);
  PybindSpeakerEmbeddingManager(&m);
  PybindSpokenLanguageIdentification(&m);

#if SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION == 1
  PybindFastClustering(&m);
  PybindOfflineSpeakerDiarizationResult(&m);
  PybindOfflineSpeakerDiarization(&m);
#else
  /* Define "empty" diarization sybmbols */
  m.attr("FastClusteringConfig") = py::none();
  m.attr("FastClustering") = py::none();
  m.attr("OfflineSpeakerDiarizationSegment") = py::none();
  m.attr("OfflineSpeakerDiarizationResult") = py::none();
  m.attr("OfflineSpeakerSegmentationPyannoteModelConfig") = py::none();
  m.attr("OfflineSpeakerSegmentationModelConfig") = py::none();
  m.attr("OfflineSpeakerDiarizationConfig") = py::none();
  m.attr("OfflineSpeakerDiarization") = py::none();
#endif

  PybindAlsa(&m);
  PybindOfflineSpeechDenoiser(&m);
  PybindOfflineSourceSeparation(&m);
  PybindVersion(&m);
}

}  // namespace sherpa_onnx
