include_directories(${CMAKE_SOURCE_DIR})

pybind11_add_module(_sherpa_onnx
  circular-buffer.cc
  display.cc
  endpoint.cc
  features.cc
  offline-ctc-fst-decoder-config.cc
  offline-lm-config.cc
  offline-model-config.cc
  offline-nemo-enc-dec-ctc-model-config.cc
  offline-paraformer-model-config.cc
  offline-recognizer.cc
  offline-stream.cc
  offline-tdnn-model-config.cc
  offline-transducer-model-config.cc
  offline-tts-model-config.cc
  offline-tts-vits-model-config.cc
  offline-tts.cc
  offline-wenet-ctc-model-config.cc
  offline-whisper-model-config.cc
  offline-zipformer-ctc-model-config.cc
  online-lm-config.cc
  online-model-config.cc
  online-paraformer-model-config.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-model-config.cc
  online-wenet-ctc-model-config.cc
  online-zipformer2-ctc-model-config.cc
  sherpa-onnx.cc
  silero-vad-model-config.cc
  speaker-embedding-extractor.cc
  speaker-embedding-manager.cc
  vad-model-config.cc
  vad-model.cc
  voice-activity-detector.cc
)

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_sherpa_onnx PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()

if(NOT WIN32)
  target_link_libraries(_sherpa_onnx PRIVATE "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/sherpa_onnx/lib")
endif()

target_link_libraries(_sherpa_onnx PRIVATE sherpa-onnx-core)

install(TARGETS _sherpa_onnx
  DESTINATION ../
)
