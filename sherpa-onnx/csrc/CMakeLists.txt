include_directories(${CMAKE_SOURCE_DIR})
add_executable(sherpa-onnx2 main.cpp)

target_link_libraries(sherpa-onnx2
  onnxruntime
  kaldi-native-fbank-core
  kaldi_native_io_core
)

add_executable(sherpa-show-onnx-info show-onnx-info.cc)
target_link_libraries(sherpa-show-onnx-info onnxruntime)

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
