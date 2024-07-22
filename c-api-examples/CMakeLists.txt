include(cargs)

include_directories(${CMAKE_SOURCE_DIR})
add_executable(decode-file-c-api decode-file-c-api.c)
target_link_libraries(decode-file-c-api sherpa-onnx-c-api cargs)

if(SHERPA_ONNX_ENABLE_TTS)
  add_executable(offline-tts-c-api offline-tts-c-api.c)
  target_link_libraries(offline-tts-c-api sherpa-onnx-c-api cargs)
endif()

add_executable(spoken-language-identification-c-api spoken-language-identification-c-api.c)
target_link_libraries(spoken-language-identification-c-api sherpa-onnx-c-api)

add_executable(speaker-identification-c-api speaker-identification-c-api.c)
target_link_libraries(speaker-identification-c-api sherpa-onnx-c-api)

add_executable(streaming-hlg-decode-file-c-api streaming-hlg-decode-file-c-api.c)
target_link_libraries(streaming-hlg-decode-file-c-api sherpa-onnx-c-api)

add_executable(audio-tagging-c-api audio-tagging-c-api.c)
target_link_libraries(audio-tagging-c-api sherpa-onnx-c-api)

add_executable(add-punctuation-c-api add-punctuation-c-api.c)
target_link_libraries(add-punctuation-c-api sherpa-onnx-c-api)

add_executable(offline-stt-c-api offline-stt-c-api.c)
target_link_libraries(offline-stt-c-api sherpa-onnx-c-api)

if(SHERPA_ONNX_HAS_ALSA)
  add_subdirectory(./asr-microphone-example)
elseif((UNIX AND NOT APPLE) OR LINUX)
  message(WARNING "Not include ./asr-microphone-example since alsa is not available")
endif()
