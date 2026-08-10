if(NOT $ENV{SHERPA_ONNX_IS_USING_BUILD_WASM_SH})
  message(FATAL_ERROR "Please use ./build-wasm-simd-web.sh to build for wasm Web")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../wasm-common.cmake)

# Browser specific flags
string(APPEND MY_FLAGS " -s 'EXPORT_NAME=\"SherpaOnnx\"' ")

message(STATUS "MY_FLAGS: ${MY_FLAGS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MY_FLAGS}")
set(CMAKE_EXECUTABLE_LINKER_FLAGS "${CMAKE_EXECUTABLE_LINKER_FLAGS} ${MY_FLAGS}")

add_executable(sherpa-onnx-wasm-web ${CMAKE_CURRENT_SOURCE_DIR}/../nodejs/sherpa-onnx-wasm-nodejs.cc)
target_link_libraries(sherpa-onnx-wasm-web sherpa-onnx-core sherpa-onnx-c-api)
install(TARGETS sherpa-onnx-wasm-web DESTINATION bin/wasm/web)

install(
  FILES
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-web>/sherpa-onnx-wasm-web.js"
    "$<TARGET_FILE_DIR:sherpa-onnx-wasm-web>/sherpa-onnx-wasm-web.wasm"
  DESTINATION
    bin/wasm/web
)
