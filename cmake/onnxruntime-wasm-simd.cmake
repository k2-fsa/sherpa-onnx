# Copyright (c)  2022-2024  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT SHERPA_ONNX_ENABLE_WASM)
  message(FATAL_ERROR "This file is for WebAssembly.")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "BUILD_SHARED_LIBS should be OFF for WebAssembly")
endif()

set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.17.1/onnxruntime-wasm-static_lib-simd-1.17.1.zip")
set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-wasm-static_lib-simd-1.17.1.zip")
set(onnxruntime_HASH "SHA256=8f07778e4233cf5a61a9d0795d90c5497177fbe8a46b701fda2d8d4e2b11cef8")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-wasm-static_lib-simd-1.17.1.zip
  ${CMAKE_SOURCE_DIR}/onnxruntime-wasm-static_lib-simd-1.17.1.zip
  ${CMAKE_BINARY_DIR}/onnxruntime-wasm-static_lib-simd-1.17.1.zip
  /tmp/onnxruntime-wasm-static_lib-simd-1.17.1.zip
  /star-fj/fangjun/download/github/onnxruntime-wasm-static_lib-simd-1.17.1.zip
)

foreach(f IN LISTS possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL  "${f}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    set(onnxruntime_URL2)
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
    ${onnxruntime_URL2}
  URL_HASH          ${onnxruntime_HASH}
)

FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
  message(STATUS "Downloading onnxruntime from ${onnxruntime_URL}")
  FetchContent_Populate(onnxruntime)
endif()
message(STATUS "onnxruntime is downloaded to ${onnxruntime_SOURCE_DIR}")

# for static libraries, we use onnxruntime_lib_files directly below
include_directories(${onnxruntime_SOURCE_DIR}/include)

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/lib*.a")

set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)

message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
