if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm-simd-nodejs.sh to build for wasm NodeJS")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../wasm-common.cmake)

# Node.js specific flags
string(APPEND MY_FLAGS " -sNODERAWFS=1 ")
string(APPEND MY_FLAGS " -sWASM_ASYNC_COMPILATION=0 ")

message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTABLE_LINKER_FLAGS "${CMAKE_EXECUTABLE_LINKER_FLAGS} ${MY_FLAGS}")

add_executable(sherpa-onnx-wasm-nodejs sherpa-onnx-wasm-nodejs.cc)
target_link_libraries(sherpa-onnx-wasm-nodejs sherpa-onnx-core sherpa-onnx-c-api)
install(TARGETS sherpa-onnx-wasm-nodejs DESTINATION bin/wasm/nodejs)

install(
  FILES
  ${CMAKE_SOURCE_DIR}/wasm/asr/sherpa-onnx-asr.js
  ${CMAKE_SOURCE_DIR}/wasm/tts/sherpa-onnx-tts.js
  ${CMAKE_SOURCE_DIR}/wasm/kws/sherpa-onnx-kws.js
  ${CMAKE_SOURCE_DIR}/wasm/vad/sherpa-onnx-vad.js
  ${CMAKE_SOURCE_DIR}/wasm/nodejs/sherpa-onnx-punctuation.js
  ${CMAKE_SOURCE_DIR}/wasm/speaker-diarization/sherpa-onnx-speaker-diarization.js
  ${CMAKE_SOURCE_DIR}/wasm/speech-enhancement/sherpa-onnx-speech-enhancement.js
  ${CMAKE_SOURCE_DIR}/wasm/nodejs/sherpa-onnx-wave.js
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-nodejs>/sherpa-onnx-wasm-nodejs.js"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-nodejs>/sherpa-onnx-wasm-nodejs.wasm"
  DESTINATION
    bin/wasm/nodejs
)
