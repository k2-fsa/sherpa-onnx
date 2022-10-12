include_directories(${CMAKE_SOURCE_DIR})
add_executable(sherpa-onnx main.cpp)

target_link_libraries(sherpa-onnx
  onnxruntime
  kaldi-native-fbank-core
  kaldi_native_io_core
)
