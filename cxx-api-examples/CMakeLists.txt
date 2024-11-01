include_directories(${CMAKE_SOURCE_DIR})

add_executable(streaming-zipformer-cxx-api ./streaming-zipformer-cxx-api.cc)
target_link_libraries(streaming-zipformer-cxx-api sherpa-onnx-cxx-api)

add_executable(streaming-zipformer-rtf-cxx-api ./streaming-zipformer-rtf-cxx-api.cc)
target_link_libraries(streaming-zipformer-rtf-cxx-api sherpa-onnx-cxx-api)

add_executable(whisper-cxx-api ./whisper-cxx-api.cc)
target_link_libraries(whisper-cxx-api sherpa-onnx-cxx-api)

add_executable(moonshine-cxx-api ./moonshine-cxx-api.cc)
target_link_libraries(moonshine-cxx-api sherpa-onnx-cxx-api)

add_executable(sense-voice-cxx-api ./sense-voice-cxx-api.cc)
target_link_libraries(sense-voice-cxx-api sherpa-onnx-cxx-api)
