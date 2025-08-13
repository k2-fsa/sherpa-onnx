include_directories(${PROJECT_SOURCE_DIR})

set(srcs
  audio-tagging.cc
  circular-buffer.cc
  cuda-config.cc
  display.cc
  endpoint.cc
  features.cc
  homophone-replacer.cc
  keyword-spotter.cc
  offline-canary-model-config.cc
  offline-ctc-fst-decoder-config.cc
  offline-dolphin-model-config.cc
  offline-fire-red-asr-model-config.cc
  offline-lm-config.cc
  offline-model-config.cc
  offline-moonshine-model-config.cc
  offline-nemo-enc-dec-ctc-model-config.cc
  offline-paraformer-model-config.cc
  offline-punctuation.cc
  offline-recognizer.cc
  offline-sense-voice-model-config.cc
  offline-source-separation-model-config.cc
  offline-source-separation-spleeter-model-config.cc
  offline-source-separation-uvr-model-config.cc
  offline-source-separation.cc
  offline-speech-denoiser-gtcrn-model-config.cc
  offline-speech-denoiser-model-config.cc
  offline-speech-denoiser.cc
  offline-stream.cc
  offline-tdnn-model-config.cc
  offline-transducer-model-config.cc
  offline-wenet-ctc-model-config.cc
  offline-whisper-model-config.cc
  offline-zipformer-ctc-model-config.cc
  online-ctc-fst-decoder-config.cc
  online-lm-config.cc
  online-model-config.cc
  online-nemo-ctc-model-config.cc
  online-paraformer-model-config.cc
  online-punctuation.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-model-config.cc
  online-wenet-ctc-model-config.cc
  online-zipformer2-ctc-model-config.cc
  provider-config.cc
  sherpa-onnx.cc
  silero-vad-model-config.cc
  speaker-embedding-extractor.cc
  speaker-embedding-manager.cc
  spoken-language-identification.cc
  ten-vad-model-config.cc
  tensorrt-config.cc
  vad-model-config.cc
  vad-model.cc
  version.cc
  voice-activity-detector.cc
  wave-writer.cc
)
if(SHERPA_ONNX_HAS_ALSA)
  list(APPEND srcs ${PROJECT_SOURCE_DIR}/sherpa-onnx/csrc/alsa.cc alsa.cc)
else()
  list(APPEND srcs faked-alsa.cc)
endif()

if(SHERPA_ONNX_ENABLE_TTS)
  list(APPEND srcs
    offline-tts-kokoro-model-config.cc
    offline-tts-matcha-model-config.cc
    offline-tts-model-config.cc
    offline-tts-vits-model-config.cc
    offline-tts.cc
  )
endif()

if(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION)
  list(APPEND srcs
    fast-clustering.cc
    offline-speaker-diarization-result.cc
    offline-speaker-diarization.cc
  )
endif()

pybind11_add_module(_sherpa_onnx ${srcs})

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  if(PYTHON_SITE_PACKAGE_DIR STREQUAL "")
    message(WARNING "PYTHON_SITE_PACKAGE_DIR is empty!")
  else()
    target_link_libraries(_sherpa_onnx PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
  endif()
endif()

if(NOT WIN32)
  target_link_libraries(_sherpa_onnx PRIVATE "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/sherpa_onnx/lib")
endif()

target_link_libraries(_sherpa_onnx PRIVATE sherpa-onnx-core)

if(SHERPA_ONNX_HAS_ALSA)
  if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
    target_link_libraries(_sherpa_onnx PRIVATE -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(_sherpa_onnx PRIVATE asound)
  endif()
endif()

install(TARGETS _sherpa_onnx
  DESTINATION ../
)
