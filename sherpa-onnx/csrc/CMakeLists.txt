include_directories(${CMAKE_SOURCE_DIR})

add_executable(sherpa-onnx
  decode.cc
  rnnt-model.cc
  sherpa-onnx.cc
  symbol-table.cc
  wave-reader.cc
)

target_link_libraries(sherpa-onnx
  onnxruntime
  kaldi-native-fbank-core
)

add_executable(sherpa-show-onnx-info show-onnx-info.cc)
target_link_libraries(sherpa-show-onnx-info onnxruntime)
