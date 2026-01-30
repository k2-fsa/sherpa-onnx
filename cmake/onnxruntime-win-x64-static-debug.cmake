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


if(CMAKE_BUILD_TYPE STREQUAL Debug)
  if(SHERPA_ONNX_USE_STATIC_CRT)
    # MT
    set(onnxruntime_HASH "SHA256=efd7c3aa9fa10a380e5534ead76627790dd533142307e3fd1de2d1fba533dd90")
  else()
    # MD
    set(onnxruntime_HASH "SHA256=68aa603aa25fd1cbe7ebef465395d0b685aa66fc8fd2df0b6d6f5a1e88621c60")
  endif()
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  if(SHERPA_ONNX_USE_STATIC_CRT)
    # MT
    set(onnxruntime_HASH "SHA256=4cf1733121eee79c9f18b048d1f5e9603079931e62af1c878c0d873ecd48900e")
  else()
    # MD
    set(onnxruntime_HASH "SHA256=ba5ae7bf3b5a29ea348f38516e7c46ff49921eb2a2e81e391f36bc932c4a7a20")
  endif()
elseif(CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
  if(SHERPA_ONNX_USE_STATIC_CRT)
    # MT
    set(onnxruntime_HASH "SHA256=2d362a781ff98731423688ff5a50a08e1dd0e863e2de5b1d66c6595945a60735")
  else()
    # MD
    set(onnxruntime_HASH "SHA256=e57978b5811fcf795e07c33eb69f32fac5cac8b848d32acf1154ce13c9cbcfd7")
  endif()
else()
  message(FATAL_ERROR "This file is for building a debug version on Windows x64. Given ${CMAKE_BUILD_TYPE}")
endif()

if(SHERPA_ONNX_USE_STATIC_CRT)
  set(onnxruntime_crt "MT")
else()
  set(onnxruntime_crt "MD")
endif()

message(STATUS "Use MSVC CRT: ${onnxruntime_crt}")

set(onnxruntime_URL  "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.23.2/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.23.2.tar.bz2")

# If you don't have access to the Internet,
# please download onnxruntime to one of the following locations.
# You can add more if you want.
set(possible_file_locations
  $ENV{HOME}/Downloads/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.23.2.tar.bz2
  ${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.23.2.tar.bz2
  ${CMAKE_BINARY_DIR}/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.23.2.tar.bz2
  /tmp/onnxruntime-win-x64-static_lib-${onnxruntime_crt}-${CMAKE_BUILD_TYPE}-1.23.2.tar.bz2
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
