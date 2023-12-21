include_directories(${CMAKE_SOURCE_DIR})
add_library(sherpa-onnx-c-api c-api.cc)

if(SHERPA_ONNX_ENABLE_TTS)
  target_sources(sherpa-onnx-c-api PRIVATE c-api-tts.cc)
endif()

target_link_libraries(sherpa-onnx-c-api sherpa-onnx-core)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(sherpa-onnx-c-api PRIVATE SHERPA_ONNX_BUILD_SHARED_LIBS=1)
  target_compile_definitions(sherpa-onnx-c-api PRIVATE SHERPA_ONNX_BUILD_MAIN_LIB=1)
endif()

if(SHERPA_ONNX_ENABLE_TTS)
  target_compile_definitions(sherpa-onnx-c-api PUBLIC SHERPA_ONNX_DISABLE_TTS=1)
endif()

install(TARGETS sherpa-onnx-c-api DESTINATION lib)

install(FILES c-api.h
  DESTINATION include/sherpa-onnx/c-api
)
