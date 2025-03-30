# Copyright (c)  2022-2024  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Linux)
  message(FATAL_ERROR "This file is for Linux only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
  message(FATAL_ERROR "This file is for aarch64 only. Given: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building shared libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

if(NOT SHERPA_ONNX_ENABLE_GPU)
  message(FATAL_ERROR "This file is for NVIDIA GPU only. Given SHERPA_ONNX_ENABLE_GPU: ${SHERPA_ONNX_ENABLE_GPU}")
endif()

message(WARNING "\
SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION: ${SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION}
If you use Jetson nano b01, then please pass
   -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.11.0
to cmake (You need to make sure CUDA 10.2 is available on your board).

If you use Jetson Orin NX, then please pass
   -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.16.0
to cmake (You need to make sure CUDA 11.4 is available on your board).

If you use NVIDIA Jetson Orin Nano Engineering Reference Developer Kit
Super - Jetpack 6.2 [L4T 36.4.3], then please pass
   -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.18.1
to cmake (You need to make sure CUDA 12.6 is available on your board).
")

set(v ${SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION})

set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v${v}/onnxruntime-linux-aarch64-gpu-${v}.tar.bz2")
set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-linux-aarch64-gpu-${v}.tar.bz2")

if(v STREQUAL "1.11.0")
  set(onnxruntime_HASH "SHA256=36eded935551e23aead09d4173bdf0bd1e7b01fdec15d77f97d6e34029aa60d7")
elseif(v STREQUAL "1.16.0")
  set(onnxruntime_HASH "SHA256=4c09d5acf2c2682b4eab1dc2f1ad98fc1fde5f5f1960063e337983ba59379a4b")
elseif(v STREQUAL "1.18.0")
  set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.18.0/onnxruntime-linux-aarch64-gpu-cuda12.2-cudnn8.9.4-trt8.6.2-1.18.0.tar.bz2")
  set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-linux-aarch64-gpu-cuda12.2-cudnn8.9.4-trt8.6.2-1.18.0.tar.bz2")
  set(onnxruntime_HASH "SHA256=da437a69be982fc28ca7d60d0c5ccce2f48d027fa888cc76458cdc05410f4e2d")
elseif(v STREQUAL "1.18.1")
  set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.18.1/onnxruntime-linux-aarch64-gpu-cuda12-1.18.1.tar.bz2")
  set(onnxruntime_URL2 "https://hf-mirror.com/csukuangfj/onnxruntime-libs/resolve/main/onnxruntime-linux-aarch64-gpu-cuda12-1.18.1.tar.bz2")
  set(onnxruntime_HASH "SHA256=1e91064ec13a6fabb6b670da8a2da4f369c1dbd50a5be77a879b2473e7afc0a6")
else()
  message(FATAL_ERROR "Unuspported onnxruntime version ${v} for Linux aarch64")
endif()

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-linux-aarch64-gpu-${v}.tar.bz2
  ${CMAKE_SOURCE_DIR}/onnxruntime-linux-aarch64-gpu-${v}.tar.bz2
  ${CMAKE_BINARY_DIR}/onnxruntime-linux-aarch64-gpu-${v}.tar.bz2
  /tmp/onnxruntime-linux-aarch64-gpu-${v}.tar.bz2
  /star-fj/fangjun/download/github/onnxruntime-linux-aarch64-gpu-${v}.tar.bz2
  #
  $ENV{HOME}/Downloads/onnxruntime-linux-aarch64-gpu-cuda12.2-cudnn8.9.4-trt8.6.2-${v}.tar.bz2
  ${CMAKE_SOURCE_DIR}/onnxruntime-linux-aarch64-gpu-cuda12.2-cudnn8.9.4-trt8.6.2-${v}.tar.bz2
  ${CMAKE_BINARY_DIR}/onnxruntime-linux-aarch64-gpu-cuda12.2-cudnn8.9.4-trt8.6.2-${v}.tar.bz2
  /tmp/onnxruntime-linux-aarch64-gpu-cuda12.2-cudnn8.9.4-trt8.6.2-${v}.tar.bz2
  /star-fj/fangjun/download/github/onnxruntime-linux-aarch64-gpu-cuda12.2-cudnn8.9.4-trt8.6.2-${v}.tar.bz2
  #
  $ENV{HOME}/Downloads/onnxruntime-linux-aarch64-gpu-cuda12-${v}.tar.bz2
  ${CMAKE_SOURCE_DIR}/onnxruntime-linux-aarch64-gpu-cuda12-${v}.tar.bz2
  ${CMAKE_BINARY_DIR}/onnxruntime-linux-aarch64-gpu-cuda12-${v}.tar.bz2
  /tmp/onnxruntime-linux-aarch64-gpu-cuda12-${v}.tar.bz2
  /star-fj/fangjun/download/github/onnxruntime-linux-aarch64-gpu-cuda12-${v}.tar.bz2
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

find_library(location_onnxruntime onnxruntime
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)

message(STATUS "location_onnxruntime: ${location_onnxruntime}")

add_library(onnxruntime SHARED IMPORTED)

set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime}
  INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
)

find_library(location_onnxruntime_cuda_lib onnxruntime_providers_cuda
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)

add_library(onnxruntime_providers_cuda SHARED IMPORTED)
set_target_properties(onnxruntime_providers_cuda PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_cuda_lib}
)
message(STATUS "location_onnxruntime_cuda_lib: ${location_onnxruntime_cuda_lib}")

# for libonnxruntime_providers_shared.so
find_library(location_onnxruntime_providers_shared_lib onnxruntime_providers_shared
  PATHS
  "${onnxruntime_SOURCE_DIR}/lib"
  NO_CMAKE_SYSTEM_PATH
)
add_library(onnxruntime_providers_shared SHARED IMPORTED)
set_target_properties(onnxruntime_providers_shared PROPERTIES
  IMPORTED_LOCATION ${location_onnxruntime_providers_shared_lib}
)
message(STATUS "location_onnxruntime_providers_shared_lib: ${location_onnxruntime_providers_shared_lib}")

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*")
message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
install(FILES ${onnxruntime_lib_files} DESTINATION lib)
