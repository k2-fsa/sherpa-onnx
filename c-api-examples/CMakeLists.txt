include(cargs)

include_directories(${CMAKE_SOURCE_DIR})
add_executable(decode-file-c-api decode-file-c-api.c)
target_link_libraries(decode-file-c-api sherpa-onnx-c-api cargs)

add_executable(offline-tts-c-api offline-tts-c-api.c)
target_link_libraries(offline-tts-c-api sherpa-onnx-c-api cargs)
