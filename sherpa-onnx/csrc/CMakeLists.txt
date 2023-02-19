include_directories(${CMAKE_SOURCE_DIR})

add_executable(sherpa-onnx
  features.cc
  online-lstm-transducer-model.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  onnx-utils.cc
  sherpa-onnx.cc
  symbol-table.cc
  wave-reader.cc
)

target_link_libraries(sherpa-onnx
  onnxruntime
  kaldi-native-fbank-core
)

add_executable(sherpa-onnx-show-info show-onnx-info.cc)
target_link_libraries(sherpa-onnx-show-info onnxruntime)
