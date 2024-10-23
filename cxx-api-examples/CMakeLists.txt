include_directories(${CMAKE_SOURCE_DIR})

add_executable(streaming-zipformer-cxx-api ./streaming-zipformer-cxx-api.cc)
target_link_libraries(streaming-zipformer-cxx-api sherpa-onnx-cxx-api)
