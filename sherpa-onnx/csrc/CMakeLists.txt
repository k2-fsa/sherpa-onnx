include_directories(${CMAKE_SOURCE_DIR})

add_library(sherpa-onnx-core
  features.cc
  online-lstm-transducer-model.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  onnx-utils.cc
  symbol-table.cc
  wave-reader.cc
)

target_link_libraries(sherpa-onnx-core
  onnxruntime
  kaldi-native-fbank-core
)

add_executable(sherpa-onnx sherpa-onnx.cc)

target_link_libraries(sherpa-onnx sherpa-onnx-core)
if(NOT WIN32)
  target_link_libraries(sherpa-onnx "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
endif()

install(TARGETS sherpa-onnx-core DESTINATION lib)
install(TARGETS sherpa-onnx DESTINATION bin)
