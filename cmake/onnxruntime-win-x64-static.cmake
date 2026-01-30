# Copyright (c)  2022-2023  Xiaomi Corporation
message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_VS_PLATFORM_NAME: ${CMAKE_VS_PLATFORM_NAME}")

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  message(FATAL_ERROR "This file is for Windows only. Given: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT (CMAKE_VS_PLATFORM_NAME STREQUAL X64 OR CMAKE_VS_PLATFORM_NAME STREQUAL x64))
  message(FATAL_ERROR "This file is for Windows x64 only. Given: ${CMAKE_VS_PLATFORM_NAME}")
endif()

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR "This file is for building static libraries. BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
endif()

if(NOT CMAKE_BUILD_TYPE STREQUAL Release)
  message(FATAL_ERROR "This file is for building a release version on Windows x64")
endif()

# Determine which CRT flavor to use
# SHERPA_ONNX_USE_STATIC_CRT = ON -> MT
# SHERPA_ONNX_USE_STATIC_CRT = OFF -> MD
if(SHERPA_ONNX_USE_STATIC_CRT)
  set(onnxruntime_crt "MT")
  set(onnxruntime_HASH "SHA256=7e19865adc0d6486089638a7431d977a62a02109a8c8cee4b6884b8ba104c193")
else()
  set(onnxruntime_crt "MD")
  set(onnxruntime_HASH "SHA256=1236aeed8aa7f53ec40212ac105d2e2d242c69c85e5bd7314a5518e70134fd32")
endif()

message(STATUS "Use MSVC CRT: ${onnxruntime_crt}")

set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-1.23.2.tar.bz2")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-1.23.2.tar.bz2
  ${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-1.23.2.tar.bz2
  ${CMAKE_BINARY_DIR}/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-1.23.2.tar.bz2
  /tmp/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-1.23.2.tar.bz2
)

foreach(f IN LISTS possible_file_locations)
  if(EXISTS ${f})
    set(onnxruntime_URL  "${f}")
    file(TO_CMAKE_PATH "${onnxruntime_URL}" onnxruntime_URL)
    message(STATUS "Found local downloaded onnxruntime: ${onnxruntime_URL}")
    break()
  endif()
endforeach()

FetchContent_Declare(onnxruntime
  URL
    ${onnxruntime_URL}
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

file(GLOB onnxruntime_lib_files "${onnxruntime_SOURCE_DIR}/lib/*.lib")

set(onnxruntime_lib_files ${onnxruntime_lib_files} PARENT_SCOPE)

message(STATUS "onnxruntime lib files: ${onnxruntime_lib_files}")
if(SHERPA_ONNX_ENABLE_PYTHON)
  install(FILES ${onnxruntime_lib_files} DESTINATION ..)
else()
  install(FILES ${onnxruntime_lib_files} DESTINATION lib)
endif()
